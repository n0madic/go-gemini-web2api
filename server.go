package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/n0madic/go-gemini-web2api/pkg/apiconv"
	"github.com/n0madic/go-gemini-web2api/pkg/gemini"
	"github.com/n0madic/go-gemini-web2api/pkg/util"
)

// maxBodyBytes caps the request body size accepted from clients.
const maxBodyBytes = 20 << 20 // 20 MiB

// googleModelRe extracts the model name from /v1beta/models/{model}:method paths.
var googleModelRe = regexp.MustCompile(`/v1beta/models/([^:?]+)`)

// Server routes HTTP requests to the Gemini client.
type Server struct {
	cfg    *Config
	client *gemini.Client
	log    *slog.Logger
}

func newServer(cfg *Config, client *gemini.Client, logger *slog.Logger) *Server {
	return &Server{cfg: cfg, client: client, log: logger}
}

// handler builds the routed, middleware-wrapped HTTP handler. Health and model
// listings are open; generation endpoints are wrapped in the auth middleware.
func (s *Server) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", s.handleHealth)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /v1beta/models", s.handleModelsGoogle)
	mux.Handle("POST /v1/chat/completions", s.withAuth(s.handleChat))
	mux.Handle("POST /v1/responses", s.withAuth(s.handleResponses))
	mux.Handle("POST /v1/messages", s.withAuth(s.handleMessages))
	mux.Handle("POST /v1/messages/count_tokens", s.withAuth(s.handleCountTokens))
	mux.Handle("POST /v1beta/models/{model}", s.withAuth(s.handleGoogleGenerate))
	mux.HandleFunc("/", s.handleNotFound)
	return s.withCORS(s.withLogging(mux))
}

// withCORS sets permissive CORS headers and answers preflight requests.
func (s *Server) withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		if r.Method == http.MethodOptions {
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "*")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// withLogging logs each request at Info level (suppressed unless GEMINI_LOG_REQUESTS).
func (s *Server) withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.log.Info("request", "method", r.Method, "path", r.URL.Path)
		next.ServeHTTP(w, r)
	})
}

// withAuth enforces the optional API-key gate around a generation handler.
func (s *Server) withAuth(next http.HandlerFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !s.authorized(r) {
			sendError(w, http.StatusUnauthorized, "missing or invalid API key")
			return
		}
		next(w, r)
	})
}

func (s *Server) handleNotFound(w http.ResponseWriter, _ *http.Request) {
	sendError(w, http.StatusNotFound, "not found")
}

// authorized reports whether the request may access generation endpoints.
func (s *Server) authorized(r *http.Request) bool {
	if len(s.cfg.APIKeys) == 0 {
		return true
	}
	key := strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "))
	if key == "" {
		key = r.Header.Get("x-api-key")
	}
	if key == "" {
		key = r.Header.Get("x-goog-api-key")
	}
	if key == "" {
		key = r.URL.Query().Get("key")
	}
	for _, allowed := range s.cfg.APIKeys {
		if subtle.ConstantTimeCompare([]byte(allowed), []byte(key)) == 1 {
			return true
		}
	}
	return false
}

// ─── Simple endpoints ────────────────────────────────────────────────────────

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	sendJSON(w, http.StatusOK, healthResponse{
		Status:  "ok",
		Version: version,
		Models:  s.client.ModelNames(),
	})
}

// handleModels serves the model list in OpenAI format by default, or Anthropic
// format when the request carries an anthropic-version header (Claude Code etc.).
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("anthropic-version") != "" {
		s.handleModelsAnthropic(w)
		return
	}
	s.handleModelsOpenAI(w)
}

func (s *Server) handleModelsOpenAI(w http.ResponseWriter) {
	sendJSON(w, http.StatusOK, apiconv.OpenAIModelList(s.client.ListModels()))
}

func (s *Server) handleModelsGoogle(w http.ResponseWriter, _ *http.Request) {
	sendJSON(w, http.StatusOK, apiconv.GoogleModelList(s.client.ListModels()))
}

func (s *Server) handleModelsAnthropic(w http.ResponseWriter) {
	sendJSON(w, http.StatusOK, apiconv.AnthropicModelList(s.client.ListModels()))
}

// ─── OpenAI Chat Completions ─────────────────────────────────────────────────

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	body, err := readBody(w, r)
	if err != nil {
		sendError(w, bodyErrorStatus(err), "invalid body: "+err.Error())
		return
	}
	var req apiconv.ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		sendError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	m, err := s.client.ResolveModel(req.Model)
	if err != nil {
		sendError(w, http.StatusBadRequest, err.Error())
		return
	}

	prompt, images, toolsActive := req.Prompt()
	if len(images) > 0 && !s.client.ImagesSupported() {
		sendError(w, http.StatusBadRequest, apiconv.ImagesNeedAuthMsg)
		return
	}
	if strings.TrimSpace(prompt) == "" && len(images) == 0 {
		sendError(w, http.StatusBadRequest, "empty prompt")
		return
	}

	cid := apiconv.ID("chatcmpl-", 12)
	p := m.Params(prompt)
	p.Images = images

	// True streaming path (no active tools): forward deltas as they arrive.
	if req.Stream && !toolsActive {
		s.streamChat(r.Context(), w, cid, m.Name, p)
		return
	}

	// Blocking path (also used for tool calling, which needs the full response).
	text, err := s.client.Generate(r.Context(), p)
	if err != nil {
		sendError(w, http.StatusBadGateway, "upstream error: "+err.Error())
		return
	}
	var toolCalls []apiconv.ToolCall
	if toolsActive && text != "" {
		text, toolCalls = apiconv.ParseToolCalls(text)
	}

	if req.Stream {
		// Tools + stream: emit one chunk carrying the full message, then DONE.
		f, ok := openSSE(w)
		if !ok {
			sendError(w, http.StatusInternalServerError, "streaming unsupported")
			return
		}
		writeSSEData(w, f, apiconv.ChatChunkComplete(cid, m.Name, text, toolCalls))
		writeSSEDone(w, f)
		return
	}

	sendJSON(w, http.StatusOK, apiconv.ChatCompletion(cid, m.Name, prompt, text, toolCalls))
}

// streamChat performs true token streaming for a chat completion.
func (s *Server) streamChat(ctx context.Context, w http.ResponseWriter, cid, model string, p gemini.GenParams) {
	created := time.Now().Unix()
	var f http.Flusher
	headerWritten := false

	emit := func(text string) {
		if !headerWritten {
			fl, ok := openSSE(w)
			if !ok {
				return
			}
			f = fl
			headerWritten = true
		}
		writeSSEData(w, f, apiconv.ChatChunk(cid, created, model, text))
	}

	err := s.client.GenerateStream(ctx, p, emit)
	if !headerWritten {
		// Nothing was streamed yet, so a normal HTTP error status is still possible.
		// No deltas and no error means the upstream produced no content.
		if err != nil {
			sendError(w, http.StatusBadGateway, "upstream error: "+err.Error())
		} else {
			sendError(w, http.StatusBadGateway, gemini.ErrEmptyResponse.Error())
		}
		return
	}
	if err != nil {
		// The stream was interrupted after deltas were already sent; the status is
		// committed, so surface the failure in-band as an error frame instead of a
		// clean "stop" chunk that would read as a complete response.
		s.log.Warn("stream interrupted", "err", err)
		writeSSEData(w, f, errorObj("upstream error: "+err.Error()))
		writeSSEDone(w, f)
		return
	}
	writeSSEData(w, f, apiconv.ChatChunkStop(cid, created, model))
	writeSSEDone(w, f)
}

// ─── OpenAI Responses API ────────────────────────────────────────────────────

func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	body, err := readBody(w, r)
	if err != nil {
		sendError(w, bodyErrorStatus(err), "invalid body: "+err.Error())
		return
	}
	var req apiconv.ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		sendError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	m, err := s.client.ResolveModel(req.Model)
	if err != nil {
		sendError(w, http.StatusBadRequest, err.Error())
		return
	}

	prompt, images, toolsActive := req.Prompt()
	if len(images) > 0 && !s.client.ImagesSupported() {
		sendError(w, http.StatusBadRequest, apiconv.ImagesNeedAuthMsg)
		return
	}
	if strings.TrimSpace(prompt) == "" && len(images) == 0 {
		sendError(w, http.StatusBadRequest, "empty input")
		return
	}

	p := m.Params(prompt)
	p.Images = images
	text, err := s.client.Generate(r.Context(), p)
	if err != nil {
		sendError(w, http.StatusBadGateway, "upstream error: "+err.Error())
		return
	}
	var toolCalls []apiconv.ToolCall
	if toolsActive && text != "" {
		text, toolCalls = apiconv.ParseToolCalls(text)
	}

	rid := apiconv.ID("resp_", 16)
	mid := apiconv.ID("msg_", 12)

	if req.Stream {
		f, ok := openSSE(w)
		if !ok {
			sendError(w, http.StatusInternalServerError, "streaming unsupported")
			return
		}
		writeSSEEvent(w, f, "response.created", apiconv.ResponseCreated(rid, m.Name))
		for _, tc := range toolCalls {
			writeSSEEvent(w, f, "response.function_call_arguments.done", apiconv.ResponseFunctionCallDone(tc))
		}
		if text != "" || len(toolCalls) == 0 {
			writeSSEEvent(w, f, "response.output_text.done", apiconv.ResponseOutputTextDone(mid, 0, text))
		}
		writeSSEEvent(w, f, "response.completed", apiconv.ResponseCompleted(rid, mid, m.Name, prompt, text, toolCalls))
		return
	}

	sendJSON(w, http.StatusOK, apiconv.ResponseObject(rid, mid, m.Name, prompt, text, toolCalls))
}

// ─── Google native API ───────────────────────────────────────────────────────

func (s *Server) handleGoogleGenerate(w http.ResponseWriter, r *http.Request) {
	body, err := readBody(w, r)
	if err != nil {
		sendError(w, bodyErrorStatus(err), "invalid body: "+err.Error())
		return
	}
	name := parseGoogleModel(r.URL.Path)
	if name == "" {
		sendError(w, http.StatusBadRequest, "model not specified in path")
		return
	}
	m, err := s.client.ResolveModel(name)
	if err != nil {
		sendError(w, http.StatusBadRequest, err.Error())
		return
	}

	var req apiconv.GenerateContentRequest
	if err := json.Unmarshal(body, &req); err != nil {
		sendError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	prompt, images, hasTools := req.Prompt()
	if len(images) > 0 && !s.client.ImagesSupported() {
		sendError(w, http.StatusBadRequest, apiconv.ImagesNeedAuthMsg)
		return
	}
	if strings.TrimSpace(prompt) == "" && len(images) == 0 {
		sendError(w, http.StatusBadRequest, "empty content")
		return
	}
	stream := strings.Contains(r.URL.Path, ":streamGenerateContent")
	p := m.Params(prompt)
	p.Images = images

	// Incremental streaming (no tools): emit a chunk per delta, then a final
	// chunk with finishReason and usage.
	if stream && !hasTools {
		s.streamGoogle(r.Context(), w, m.Name, p)
		return
	}

	text, err := s.client.Generate(r.Context(), p)
	if err != nil {
		sendError(w, http.StatusBadGateway, "upstream error: "+err.Error())
		return
	}
	response := apiconv.GoogleResponse(m.Name, prompt, text, hasTools)

	if stream {
		f, ok := openSSE(w)
		if !ok {
			sendError(w, http.StatusInternalServerError, "streaming unsupported")
			return
		}
		writeSSEData(w, f, response)
		return
	}
	sendJSON(w, http.StatusOK, response)
}

// streamGoogle performs incremental streaming for a tool-free Google native request.
func (s *Server) streamGoogle(ctx context.Context, w http.ResponseWriter, model string, p gemini.GenParams) {
	var f http.Flusher
	started := false
	var full strings.Builder

	emit := func(delta string) {
		if !started {
			fl, ok := openSSE(w)
			if !ok {
				return
			}
			f = fl
			started = true
		}
		full.WriteString(delta)
		writeSSEData(w, f, apiconv.GoogleStreamChunk(model, delta))
	}

	err := s.client.GenerateStream(ctx, p, emit)
	if !started {
		if err != nil {
			sendError(w, http.StatusBadGateway, "upstream error: "+err.Error())
		} else {
			sendError(w, http.StatusBadGateway, gemini.ErrEmptyResponse.Error())
		}
		return
	}
	if err != nil {
		// Interrupted after deltas: emit a final frame with a non-STOP finishReason
		// so the client can tell the generation did not complete normally.
		s.log.Warn("stream interrupted", "err", err)
		writeSSEData(w, f, apiconv.GoogleStreamError(model, p.Prompt, full.String()))
		return
	}
	writeSSEData(w, f, apiconv.GoogleStreamFinal(model, p.Prompt, full.String()))
}

func parseGoogleModel(path string) string {
	if m := googleModelRe.FindStringSubmatch(path); len(m) >= 2 {
		return m[1]
	}
	return ""
}

// ─── Anthropic Messages API ──────────────────────────────────────────────────

func (s *Server) handleMessages(w http.ResponseWriter, r *http.Request) {
	body, err := readBody(w, r)
	if err != nil {
		sendAnthropicError(w, bodyErrorStatus(err), "invalid body: "+err.Error())
		return
	}
	var req apiconv.AnthropicRequest
	if err := json.Unmarshal(body, &req); err != nil {
		sendAnthropicError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	m := s.client.ResolveModelOrDefault(req.Model)
	prompt, images, toolsActive := req.Prompt()
	if len(images) > 0 && !s.client.ImagesSupported() {
		sendAnthropicError(w, http.StatusBadRequest, apiconv.ImagesNeedAuthMsg)
		return
	}
	if strings.TrimSpace(prompt) == "" && len(images) == 0 {
		sendAnthropicError(w, http.StatusBadRequest, "empty prompt")
		return
	}

	p := m.Params(prompt)
	p.Images = images
	msgID := apiconv.ID("msg_", 24)

	// True streaming (no active tools): forward text deltas as they arrive.
	if req.Stream && !toolsActive {
		s.streamMessages(r.Context(), w, msgID, m.Name, p)
		return
	}

	text, err := s.client.Generate(r.Context(), p)
	if err != nil {
		sendAnthropicError(w, http.StatusBadGateway, "upstream error: "+err.Error())
		return
	}
	var toolCalls []apiconv.ToolCall
	if toolsActive && text != "" {
		text, toolCalls = apiconv.ParseToolCalls(text)
	}

	if req.Stream {
		// Tools + stream: replay the full message as an SSE event sequence.
		s.streamMessagesFull(w, msgID, m.Name, prompt, text, toolCalls)
		return
	}
	sendJSON(w, http.StatusOK, apiconv.AnthropicResponse(m.Name, prompt, text, toolCalls))
}

func (s *Server) handleCountTokens(w http.ResponseWriter, r *http.Request) {
	body, err := readBody(w, r)
	if err != nil {
		sendAnthropicError(w, bodyErrorStatus(err), "invalid body: "+err.Error())
		return
	}
	var req apiconv.AnthropicRequest
	if err := json.Unmarshal(body, &req); err != nil {
		sendAnthropicError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	prompt, _, _ := req.Prompt()
	sendJSON(w, http.StatusOK, apiconv.AnthropicCountTokens(prompt))
}

// streamMessages performs true token streaming for a tool-free message request.
func (s *Server) streamMessages(ctx context.Context, w http.ResponseWriter, msgID, model string, p gemini.GenParams) {
	inputTokens := apiconv.ApproxTokens(p.Prompt)
	var f http.Flusher
	started := false
	var acc strings.Builder

	open := func() bool {
		fl, ok := openSSE(w)
		if !ok {
			return false
		}
		f = fl
		started = true
		writeSSEEvent(w, f, "message_start", apiconv.AnthropicMessageStart(msgID, model, inputTokens))
		writeSSEEvent(w, f, "content_block_start", apiconv.AnthropicTextBlockStart(0))
		return true
	}

	emit := func(text string) {
		if !started && !open() {
			return
		}
		acc.WriteString(text)
		writeSSEEvent(w, f, "content_block_delta", apiconv.AnthropicTextDelta(0, text))
	}

	err := s.client.GenerateStream(ctx, p, emit)
	if !started {
		// Nothing was streamed yet, so a normal HTTP error status is still possible.
		// No deltas and no error means the upstream produced no content.
		if err != nil {
			sendAnthropicError(w, http.StatusBadGateway, "upstream error: "+err.Error())
		} else {
			sendAnthropicError(w, http.StatusBadGateway, gemini.ErrEmptyResponse.Error())
		}
		return
	}
	if err != nil {
		// Interrupted after deltas: close the open block and emit an Anthropic
		// `error` event instead of a clean message_stop, so the partial response is
		// not presented as complete.
		s.log.Warn("stream interrupted", "err", err)
		writeSSEEvent(w, f, "content_block_stop", apiconv.AnthropicBlockStop(0))
		writeSSEEvent(w, f, "error", apiconv.AnthropicError("upstream error: "+err.Error()))
		return
	}
	writeSSEEvent(w, f, "content_block_stop", apiconv.AnthropicBlockStop(0))
	writeSSEEvent(w, f, "message_delta", apiconv.AnthropicMessageDelta("end_turn", apiconv.ApproxTokens(acc.String())))
	writeSSEEvent(w, f, "message_stop", apiconv.AnthropicMessageStop())
}

// streamMessagesFull replays an already-computed message as an SSE sequence, used
// when tools are present (the tool calls need the full text first). It mirrors the
// block layout of apiconv.AnthropicResponse: a text block (when there is text),
// then a tool_use block per call, or a single empty text block when there is
// neither.
func (s *Server) streamMessagesFull(w http.ResponseWriter, msgID, model, prompt, text string, toolCalls []apiconv.ToolCall) {
	f, ok := openSSE(w)
	if !ok {
		sendAnthropicError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	writeSSEEvent(w, f, "message_start", apiconv.AnthropicMessageStart(msgID, model, apiconv.ApproxTokens(prompt)))

	idx := 0
	emitTextBlock := func(t string) {
		writeSSEEvent(w, f, "content_block_start", apiconv.AnthropicTextBlockStart(idx))
		writeSSEEvent(w, f, "content_block_delta", apiconv.AnthropicTextDelta(idx, t))
		writeSSEEvent(w, f, "content_block_stop", apiconv.AnthropicBlockStop(idx))
		idx++
	}
	if text != "" {
		emitTextBlock(text)
	}
	for _, tc := range toolCalls {
		writeSSEEvent(w, f, "content_block_start", apiconv.AnthropicToolUseBlockStart(idx, tc))
		writeSSEEvent(w, f, "content_block_delta", apiconv.AnthropicToolUseDelta(idx, tc))
		writeSSEEvent(w, f, "content_block_stop", apiconv.AnthropicBlockStop(idx))
		idx++
	}
	if text == "" && len(toolCalls) == 0 {
		emitTextBlock("")
	}

	stop := "end_turn"
	if len(toolCalls) > 0 {
		stop = "tool_use"
	}
	writeSSEEvent(w, f, "message_delta", apiconv.AnthropicMessageDelta(stop, apiconv.ApproxTokens(text)))
	writeSSEEvent(w, f, "message_stop", apiconv.AnthropicMessageStop())
}

// sendAnthropicError writes an Anthropic-shaped error envelope.
func sendAnthropicError(w http.ResponseWriter, status int, msg string) {
	sendJSON(w, status, apiconv.AnthropicError(msg))
}

// ─── Output types (proxy-level) ──────────────────────────────────────────────

// errorResponse is the OpenAI-style error envelope.
type errorResponse struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Message string `json:"message"`
}

// healthResponse is returned by GET /.
type healthResponse struct {
	Status  string   `json:"status"`
	Version string   `json:"version"`
	Models  []string `json:"models"`
}

// ─── HTTP / SSE helpers ──────────────────────────────────────────────────────

// readBody reads the request body, capping it at maxBodyBytes. It uses
// http.MaxBytesReader so an over-limit body surfaces as an explicit
// *http.MaxBytesError (mapped to 413 by bodyErrorStatus) rather than being
// silently truncated into a confusing JSON parse error.
func readBody(w http.ResponseWriter, r *http.Request) ([]byte, error) {
	defer r.Body.Close()
	return io.ReadAll(http.MaxBytesReader(w, r.Body, maxBodyBytes))
}

// bodyErrorStatus maps a readBody error onto an HTTP status: 413 when the body
// exceeded maxBodyBytes, 400 otherwise.
func bodyErrorStatus(err error) int {
	var maxErr *http.MaxBytesError
	if errors.As(err, &maxErr) {
		return http.StatusRequestEntityTooLarge
	}
	return http.StatusBadRequest
}

// errorObj wraps a message in the OpenAI error envelope.
func errorObj(msg string) errorResponse {
	return errorResponse{Error: errorDetail{Message: msg}}
}

// sendError writes a JSON error envelope with the given status code.
func sendError(w http.ResponseWriter, status int, msg string) {
	sendJSON(w, status, errorObj(msg))
}

// sendJSON writes a JSON response with the given status code.
func sendJSON(w http.ResponseWriter, status int, v any) {
	b, err := util.MarshalNoEscape(v)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(b)
}

// openSSE writes SSE response headers and returns the flusher.
func openSSE(w http.ResponseWriter) (http.Flusher, bool) {
	f, ok := w.(http.Flusher)
	if !ok {
		return nil, false
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	f.Flush()
	return f, true
}

// writeSSEData writes a `data: <json>` SSE frame.
func writeSSEData(w http.ResponseWriter, f http.Flusher, v any) {
	b, err := util.MarshalNoEscape(v)
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
	f.Flush()
}

// writeSSEEvent writes a named SSE event with a JSON payload.
func writeSSEEvent(w http.ResponseWriter, f http.Flusher, event string, v any) {
	b, err := util.MarshalNoEscape(v)
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
	f.Flush()
}

// writeSSEDone writes the terminating [DONE] frame.
func writeSSEDone(w http.ResponseWriter, f http.Flusher) {
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	f.Flush()
}
