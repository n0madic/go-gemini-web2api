package apiconv

import (
	"crypto/rand"
	"encoding/hex"
	"time"

	"github.com/n0madic/go-gemini-web2api/pkg/gemini"
)

// ImagesNeedAuthMsg is returned when a request carries images but the session is
// not authenticated (anonymous, or a cookie that no longer signs in).
const ImagesNeedAuthMsg = "image input requires an authenticated cookie; the proxy has no signed-in session (running anonymously or the cookie no longer authenticates)"

// modelCreated is the static "created" timestamp reported for every model.
const modelCreated = 1700000000

// ─── OpenAI Chat Completions builders ────────────────────────────────────────

// ChatCompletion builds a non-streaming /v1/chat/completions response.
func ChatCompletion(id, model, prompt, text string, toolCalls []ToolCall) any {
	finish := "stop"
	if len(toolCalls) > 0 {
		finish = "tool_calls"
	}
	msg := outMessage{Role: "assistant", Content: contentPtr(text), ToolCalls: toolCalls}
	return chatCompletion{
		ID: id, Object: "chat.completion", Created: now(), Model: model,
		Choices: []chatChoice{{Index: 0, Message: &msg, FinishReason: &finish}},
		Usage:   usageFor(prompt, text),
	}
}

// ChatChunk builds a streaming chat.completion.chunk carrying one content delta.
func ChatChunk(id string, created int64, model, content string) any {
	c := content
	return chatChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
		Choices: []chunkChoice{{Index: 0, Delta: chunkDelta{Content: &c}, FinishReason: nil}},
	}
}

// ChatChunkStop builds the terminating chat.completion.chunk (empty delta with a
// "stop" finish reason).
func ChatChunkStop(id string, created int64, model string) any {
	stop := "stop"
	return chatChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
		Choices: []chunkChoice{{Index: 0, Delta: chunkDelta{}, FinishReason: &stop}},
	}
}

// ChatChunkComplete builds the single chat.completion.chunk used for the
// tool-calling stream path: it carries the full assistant message and a finish
// reason in one frame.
func ChatChunkComplete(id, model, text string, toolCalls []ToolCall) any {
	finish := "stop"
	if len(toolCalls) > 0 {
		finish = "tool_calls"
	}
	msg := outMessage{Role: "assistant", Content: contentPtr(text), ToolCalls: toolCalls}
	return chatChunk{
		ID: id, Object: "chat.completion.chunk", Created: now(), Model: model,
		Choices: []chunkChoice{{Index: 0, Delta: deltaFromMsg(msg), FinishReason: &finish}},
	}
}

// OpenAIModelList builds the OpenAI GET /v1/models payload.
func OpenAIModelList(models []*gemini.AvailableModel) any {
	data := make([]modelObject, 0, len(models))
	for _, m := range models {
		data = append(data, modelObject{
			ID:          m.Name,
			Object:      "model",
			Created:     modelCreated,
			OwnedBy:     "google",
			Description: m.Description,
		})
	}
	return modelsListResponse{Object: "list", Data: data}
}

// ─── OpenAI Responses API builders ───────────────────────────────────────────

// ResponseObject builds a non-streaming /v1/responses reply.
func ResponseObject(rid, mid, model, prompt, text string, toolCalls []ToolCall) any {
	in, out := ApproxTokens(prompt), ApproxTokens(text)
	return responseObject{
		ID: rid, Object: "response", CreatedAt: now(), Status: "completed",
		Model:  model,
		Output: responsesOutput(mid, text, toolCalls),
		Usage:  &responsesUsage{InputTokens: in, OutputTokens: out, TotalTokens: in + out},
	}
}

// ResponseCreated builds the response.created event payload (in-progress, empty
// output).
func ResponseCreated(rid, model string) any {
	return responseEnvelope{
		Type: "response.created",
		Response: responseObject{
			ID: rid, Object: "response", Status: "in_progress",
			Model: model, Output: []respOutputItem{},
		},
	}
}

// ResponseFunctionCallDone builds a response.function_call_arguments.done event
// payload for a single tool call.
func ResponseFunctionCallDone(tc ToolCall) any {
	return functionCallDoneEvent{
		Type:   "response.function_call_arguments.done",
		ItemID: tc.ID, CallID: tc.ID,
		Name: tc.Function.Name, Arguments: tc.Function.Arguments,
	}
}

// ResponseOutputTextDone builds a response.output_text.done event payload.
func ResponseOutputTextDone(mid string, contentIndex int, text string) any {
	return outputTextDoneEvent{
		Type:   "response.output_text.done",
		ItemID: mid, ContentIndex: contentIndex, Text: text,
	}
}

// ResponseCompleted builds the response.completed event payload (full output +
// usage).
func ResponseCompleted(rid, mid, model, prompt, text string, toolCalls []ToolCall) any {
	in, out := ApproxTokens(prompt), ApproxTokens(text)
	return responseEnvelope{
		Type: "response.completed",
		Response: responseObject{
			ID: rid, Object: "response", Status: "completed",
			Model:  model,
			Output: responsesOutput(mid, text, toolCalls),
			Usage:  &responsesUsage{InputTokens: in, OutputTokens: out, TotalTokens: in + out},
		},
	}
}

// responsesOutput builds the Responses output item list: a function_call item per
// tool call, then a message item carrying the text (the message item is emitted
// when there is text, or when there are no tool calls at all).
func responsesOutput(mid, text string, toolCalls []ToolCall) []respOutputItem {
	var output []respOutputItem
	for _, tc := range toolCalls {
		output = append(output, respOutputItem{
			Type: "function_call", ID: tc.ID, CallID: tc.ID,
			Name: tc.Function.Name, Arguments: tc.Function.Arguments, Status: "completed",
		})
	}
	if text != "" || len(toolCalls) == 0 {
		output = append(output, respOutputItem{
			Type: "message", ID: mid, Role: "assistant", Status: "completed",
			Content: []respContent{{Type: "output_text", Text: text, Annotations: []any{}}},
		})
	}
	return output
}

// ─── Output types ────────────────────────────────────────────────────────────

type chatCompletion struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   usage        `json:"usage"`
}

type chatChoice struct {
	Index        int         `json:"index"`
	Message      *outMessage `json:"message,omitempty"`
	FinishReason *string     `json:"finish_reason"`
}

type outMessage struct {
	Role      string     `json:"role"`
	Content   *string    `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type chatChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []chunkChoice `json:"choices"`
}

type chunkChoice struct {
	Index        int        `json:"index"`
	Delta        chunkDelta `json:"delta"`
	FinishReason *string    `json:"finish_reason"`
}

// chunkDelta is a streaming delta. All fields use omitempty so a content delta
// is {"content":"..."} and the terminating delta is {}.
type chunkDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   *string    `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type respOutputItem struct {
	Type      string        `json:"type"`
	ID        string        `json:"id,omitempty"`
	CallID    string        `json:"call_id,omitempty"`
	Name      string        `json:"name,omitempty"`
	Arguments string        `json:"arguments,omitempty"`
	Status    string        `json:"status,omitempty"`
	Role      string        `json:"role,omitempty"`
	Content   []respContent `json:"content,omitempty"`
}

type respContent struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	Annotations []any  `json:"annotations"`
}

// modelsListResponse is the OpenAI GET /v1/models payload.
type modelsListResponse struct {
	Object string        `json:"object"`
	Data   []modelObject `json:"data"`
}

type modelObject struct {
	ID          string `json:"id"`
	Object      string `json:"object"`
	Created     int64  `json:"created"`
	OwnedBy     string `json:"owned_by"`
	Description string `json:"description"`
}

// responseObject is a Responses API response, used for both the non-streaming
// reply and the embedded object in response.created/response.completed events.
// CreatedAt and Usage are omitted from the in-progress/created event.
type responseObject struct {
	ID        string           `json:"id"`
	Object    string           `json:"object"`
	CreatedAt int64            `json:"created_at,omitempty"`
	Status    string           `json:"status"`
	Model     string           `json:"model"`
	Output    []respOutputItem `json:"output"`
	Usage     *responsesUsage  `json:"usage,omitempty"`
}

type responsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// responseEnvelope wraps a responseObject for response.created/completed events.
type responseEnvelope struct {
	Type     string         `json:"type"`
	Response responseObject `json:"response"`
}

type functionCallDoneEvent struct {
	Type      string `json:"type"`
	ItemID    string `json:"item_id"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type outputTextDoneEvent struct {
	Type         string `json:"type"`
	ItemID       string `json:"item_id"`
	ContentIndex int    `json:"content_index"`
	Text         string `json:"text"`
}

// ─── Shared helpers ──────────────────────────────────────────────────────────

// now returns the current Unix timestamp.
func now() int64 { return time.Now().Unix() }

// hexToken returns a random lowercase hex string of n hex characters.
// Used for short identifiers like call_xxxxxxxx and chatcmpl-xxxxxxxxxxxx.
func hexToken(n int) string {
	nbytes := (n + 1) / 2
	b := make([]byte, nbytes)
	if _, err := rand.Read(b); err != nil {
		return "00000000000000000000000000000000"[:n]
	}
	return hex.EncodeToString(b)[:n]
}

// ID returns a random response/message id with the given prefix and hex length,
// e.g. ID("chatcmpl-", 12). The HTTP layer mints one per request and threads it
// through the response and SSE-frame builders so a stream's frames share an id.
func ID(prefix string, n int) string {
	return prefix + hexToken(n)
}

// ApproxTokens estimates a token count as len/4, matching the reference.
func ApproxTokens(s string) int {
	return len([]rune(s)) / 4
}

// contentPtr returns a pointer to text, or nil when empty (so JSON emits null).
func contentPtr(text string) *string {
	if text == "" {
		return nil
	}
	return &text
}

// deltaFromMsg builds a streaming delta carrying a full assistant message,
// used for the single-chunk tool-calling stream response.
func deltaFromMsg(msg outMessage) chunkDelta {
	return chunkDelta{Role: msg.Role, Content: msg.Content, ToolCalls: msg.ToolCalls}
}

// usageFor computes approximate token usage for a prompt/completion pair.
func usageFor(prompt, completion string) usage {
	p, c := ApproxTokens(prompt), ApproxTokens(completion)
	return usage{PromptTokens: p, CompletionTokens: c, TotalTokens: p + c}
}
