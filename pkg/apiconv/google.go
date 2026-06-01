package apiconv

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/n0madic/go-gemini-web2api/pkg/gemini"
)

// ─── Google native API (/v1beta) request types ───────────────────────────────

// GenerateContentRequest is the Google AI generateContent request body.
type GenerateContentRequest struct {
	Contents          []GoogleContent   `json:"contents"`
	SystemInstruction *GoogleContent    `json:"systemInstruction"`
	Tools             []GoogleTool      `json:"tools"`
	ToolConfig        *GoogleToolConfig `json:"toolConfig"`
}

// GoogleContent is a content block with a role and parts.
type GoogleContent struct {
	Role  string       `json:"role"`
	Parts []GooglePart `json:"parts"`
}

// GooglePart is a single content part: text, inline image, function call, or
// function response.
type GooglePart struct {
	Text             string                  `json:"text,omitempty"`
	InlineData       *googleInlineData       `json:"inlineData,omitempty"`
	FunctionCall     *googleFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *googleFunctionResponse `json:"functionResponse,omitempty"`
}

// googleInlineData is an inline (base64-encoded) binary part, e.g. an image.
type googleInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

// googleFunctionCall is a model-issued tool call.
type googleFunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

// googleFunctionResponse is a client-supplied tool result.
type googleFunctionResponse struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

// GoogleTool groups function declarations.
type GoogleTool struct {
	FunctionDeclarations []GoogleFunctionDecl `json:"functionDeclarations"`
}

// GoogleFunctionDecl declares a callable function.
type GoogleFunctionDecl struct {
	Name                 string          `json:"name"`
	Description          string          `json:"description"`
	Parameters           json.RawMessage `json:"parameters"`
	ParametersJSONSchema json.RawMessage `json:"parametersJsonSchema"`
}

// GoogleToolConfig carries the function-calling mode.
type GoogleToolConfig struct {
	FunctionCallingConfig *GoogleFCConfig `json:"functionCallingConfig"`
}

// GoogleFCConfig is the functionCallingConfig (mode AUTO/ANY/NONE).
type GoogleFCConfig struct {
	Mode                 string   `json:"mode"`
	AllowedFunctionNames []string `json:"allowedFunctionNames"`
}

// fcMode returns the function-calling mode (defaults to AUTO).
func (r *GenerateContentRequest) fcMode() string {
	if r.ToolConfig != nil && r.ToolConfig.FunctionCallingConfig != nil && r.ToolConfig.FunctionCallingConfig.Mode != "" {
		return r.ToolConfig.FunctionCallingConfig.Mode
	}
	return "AUTO"
}

// hasTools reports whether the request supplies callable tools and tools are enabled.
func (r *GenerateContentRequest) hasTools() bool {
	if r.fcMode() == "NONE" {
		return false
	}
	for _, t := range r.Tools {
		if len(t.FunctionDeclarations) > 0 {
			return true
		}
	}
	return false
}

// Prompt builds the prompt, image attachments, and tool-active flag for a Google
// generateContent request.
func (r *GenerateContentRequest) Prompt() (prompt string, images []gemini.InputImage, toolsActive bool) {
	prompt = googleContentsToPrompt(r)
	images = imagesFromGoogleRequest(r)
	toolsActive = r.hasTools()
	return prompt, images, toolsActive
}

// ─── Prompt building ──────────────────────────────────────────────────────────

// googleContentsToPrompt converts a Google API request into a prompt string,
// rendering tools, function calls, and function responses into the prompt.
func googleContentsToPrompt(req *GenerateContentRequest) string {
	var parts []string

	toolDefs := googleToolDefs(req)
	sysText := ""
	if req.SystemInstruction != nil {
		if t := joinPartsText(req.SystemInstruction.Parts); t != "" {
			sysText = "[System instruction]: " + t
		}
	}

	switch {
	case sysText != "" && len(toolDefs) > 0:
		parts = append(parts, sysText+"\n\n"+buildGoogleToolPrompt(toolDefs)+googleToolChoiceInstruction(req))
	case sysText != "":
		parts = append(parts, sysText)
	case len(toolDefs) > 0:
		parts = append(parts, buildGoogleToolPrompt(toolDefs)+googleToolChoiceInstruction(req))
	}

	for _, content := range req.Contents {
		var msgParts []string
		for _, p := range content.Parts {
			switch {
			case p.Text != "":
				msgParts = append(msgParts, p.Text)
			case p.FunctionCall != nil:
				args := string(p.FunctionCall.Args)
				if args == "" {
					args = "{}"
				}
				msgParts = append(msgParts, fmt.Sprintf("```function_call\n{\"name\": %q, \"args\": %s}\n```",
					p.FunctionCall.Name, args))
			case p.FunctionResponse != nil:
				resp := string(p.FunctionResponse.Response)
				if resp == "" {
					resp = "{}"
				}
				msgParts = append(msgParts, fmt.Sprintf("[Tool result for %s]: %s", p.FunctionResponse.Name, resp))
			}
		}
		text := strings.Join(msgParts, "\n")
		if content.Role == "model" {
			parts = append(parts, "[Assistant]: "+text)
		} else {
			parts = append(parts, text)
		}
	}

	return joinNonEmpty(parts, "\n\n")
}

// googleToolDef is the normalized tool shape embedded into the prompt.
type googleToolDef struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters,omitempty"`
}

// googleToolDefs flattens functionDeclarations from the request tools.
func googleToolDefs(req *GenerateContentRequest) []googleToolDef {
	if !req.hasTools() {
		return nil
	}
	var out []googleToolDef
	for _, t := range req.Tools {
		for _, fn := range t.FunctionDeclarations {
			td := googleToolDef{Name: fn.Name, Description: fn.Description}
			params := fn.Parameters
			if len(params) == 0 {
				params = fn.ParametersJSONSchema
			}
			if len(params) > 0 {
				var p any
				if json.Unmarshal(params, &p) == nil {
					td.Parameters = p
				}
			}
			out = append(out, td)
		}
	}
	return out
}

// buildGoogleToolPrompt builds a natural tool-use prompt using the function_call
// format, designed to be followed reliably by Gemini Web.
func buildGoogleToolPrompt(toolDefs []googleToolDef) string {
	spec, _ := json.MarshalIndent(toolDefs, "", "  ")
	return "# Tool Use\n\n" +
		"You can call the following tools to help accomplish tasks. " +
		"These tools connect to the user's local environment and will execute when called.\n\n" +
		"Call format (use this exact format):\n" +
		"```function_call\n" +
		"{\"name\": \"<tool_name>\", \"args\": {<arguments>}}\n" +
		"```\n\n" +
		"When calling tools:\n" +
		"- Output ONLY the function_call block(s), nothing else\n" +
		"- You may call multiple tools with multiple blocks\n" +
		"- After receiving a [Tool result for ...], use that data to answer the user\n\n" +
		"Available tools:\n" + string(spec)
}

// googleToolChoiceInstruction renders the toolConfig constraint instruction.
func googleToolChoiceInstruction(req *GenerateContentRequest) string {
	mode := req.fcMode()
	var allowed []string
	if req.ToolConfig != nil && req.ToolConfig.FunctionCallingConfig != nil {
		allowed = req.ToolConfig.FunctionCallingConfig.AllowedFunctionNames
	}
	switch mode {
	case "NONE":
		return "\n\nIMPORTANT: Do NOT call any tools. Respond with text only."
	case "ANY":
		if len(allowed) > 0 {
			names := make([]string, len(allowed))
			for i, n := range allowed {
				names[i] = fmt.Sprintf("%q", n)
			}
			return "\n\nIMPORTANT: You MUST call one of these tools: " + strings.Join(names, ", ") +
				". Do not respond with text only."
		}
		return "\n\nIMPORTANT: You MUST call at least one tool. Do not respond with text only."
	}
	return ""
}

// joinPartsText concatenates the text of all parts with single spaces.
func joinPartsText(parts []GooglePart) string {
	var texts []string
	for _, p := range parts {
		if p.Text != "" {
			texts = append(texts, p.Text)
		}
	}
	return strings.Join(texts, " ")
}

// ─── Function-call parsing ────────────────────────────────────────────────────

var (
	googleFCBlockRe = regexp.MustCompile("(?s)```function_call\\s*\\n(.*?)\\n```")
	// Greedy to the last brace so nested objects (e.g. "args":{...}) are captured whole.
	googleFCBareRe   = regexp.MustCompile(`(?s)(?:^|\n)function_call\s*\n(\{[^` + "`" + `]*\})`)
	googleFCParsers  = []*regexp.Regexp{googleFCBlockRe, googleFCBareRe}
	googleJSONObject = regexp.MustCompile(`^\s*\{`)
)

// GoogleFunctionCall is a parsed function call (name + raw args object).
type GoogleFunctionCall struct {
	Name string
	Args json.RawMessage
}

// ParseGoogleFunctionCalls extracts function_call invocations from model output,
// handling fenced blocks, bare blocks, and a raw JSON object. Returns the cleaned
// text and the calls.
func ParseGoogleFunctionCalls(text string) (string, []GoogleFunctionCall) {
	var calls []GoogleFunctionCall
	clean := text
	for _, re := range googleFCParsers {
		for _, m := range re.FindAllStringSubmatch(clean, -1) {
			if call, ok := decodeGoogleCall(m[1]); ok {
				calls = append(calls, call)
			}
		}
		clean = strings.TrimSpace(re.ReplaceAllString(clean, ""))
	}
	if len(calls) == 0 && googleJSONObject.MatchString(clean) {
		if call, ok := decodeGoogleCall(clean); ok {
			calls = append(calls, call)
			clean = ""
		}
	}
	return clean, calls
}

// decodeGoogleCall parses a single function-call JSON object, accepting both
// "args" and "arguments" keys.
func decodeGoogleCall(s string) (GoogleFunctionCall, bool) {
	var data struct {
		Name      string          `json:"name"`
		Args      json.RawMessage `json:"args"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(s)), &data); err != nil || data.Name == "" {
		return GoogleFunctionCall{}, false
	}
	args := data.Args
	if len(args) == 0 {
		args = data.Arguments
	}
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	return GoogleFunctionCall{Name: data.Name, Args: args}, true
}

// ─── Response builders ───────────────────────────────────────────────────────

// GoogleResponse builds a complete generateContent reply (used for both the
// non-streaming reply and the single frame of a streamed tool-calling response).
// When hasTools is set and the model emitted function_call blocks, the text is
// replaced with the parsed call parts.
func GoogleResponse(model, prompt, text string, hasTools bool) any {
	parts := []GooglePart{{Text: text}}
	if hasTools && text != "" {
		clean, calls := ParseGoogleFunctionCalls(text)
		if len(calls) > 0 {
			parts = parts[:0]
			if clean != "" {
				parts = append(parts, GooglePart{Text: clean})
			}
			for _, c := range calls {
				parts = append(parts, GooglePart{FunctionCall: &googleFunctionCall{Name: c.Name, Args: c.Args}})
			}
		}
	}
	in, out := ApproxTokens(prompt), ApproxTokens(text)
	return generateContentResponse{
		Candidates: []googleCandidate{{
			Content:      &googleContentOut{Parts: parts, Role: "model"},
			FinishReason: "STOP",
			Index:        0,
		}},
		UsageMetadata: &googleUsageMetadata{
			PromptTokenCount:     in,
			CandidatesTokenCount: out,
			TotalTokenCount:      in + out,
		},
		ModelVersion: model,
	}
}

// GoogleStreamChunk builds a single incremental streamGenerateContent frame
// carrying one text delta.
func GoogleStreamChunk(model, delta string) any {
	return generateContentResponse{
		Candidates:   []googleCandidate{{Content: &googleContentOut{Parts: []GooglePart{{Text: delta}}, Role: "model"}, Index: 0}},
		ModelVersion: model,
	}
}

// GoogleStreamFinal builds the trailing streamGenerateContent frame carrying the
// finish reason and aggregate usage.
func GoogleStreamFinal(model, prompt, full string) any {
	in, out := ApproxTokens(prompt), ApproxTokens(full)
	return generateContentResponse{
		Candidates: []googleCandidate{{FinishReason: "STOP", Index: 0}},
		UsageMetadata: &googleUsageMetadata{
			PromptTokenCount:     in,
			CandidatesTokenCount: out,
			TotalTokenCount:      in + out,
		},
		ModelVersion: model,
	}
}

// GoogleStreamError builds the trailing streamGenerateContent frame for an
// interrupted generation: it reports the usage gathered so far with a non-STOP
// finishReason ("OTHER") so the client can tell the stream did not complete
// normally, instead of receiving a clean STOP that reads as a full response.
func GoogleStreamError(model, prompt, full string) any {
	in, out := ApproxTokens(prompt), ApproxTokens(full)
	return generateContentResponse{
		Candidates: []googleCandidate{{FinishReason: "OTHER", Index: 0}},
		UsageMetadata: &googleUsageMetadata{
			PromptTokenCount:     in,
			CandidatesTokenCount: out,
			TotalTokenCount:      in + out,
		},
		ModelVersion: model,
	}
}

// GoogleModelList builds the Google GET /v1beta/models payload.
func GoogleModelList(models []*gemini.AvailableModel) any {
	out := make([]googleModelObject, 0, len(models))
	for _, m := range models {
		out = append(out, googleModelObject{
			Name:                       "models/" + m.Name,
			DisplayName:                m.Name,
			Description:                m.Description,
			SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		})
	}
	return googleModelsResponse{Models: out}
}

// ─── Output types ────────────────────────────────────────────────────────────

// generateContentResponse is the Google native generateContent reply (also used
// for streamed chunks, where Content/FinishReason/UsageMetadata vary per frame).
type generateContentResponse struct {
	Candidates    []googleCandidate    `json:"candidates"`
	UsageMetadata *googleUsageMetadata `json:"usageMetadata,omitempty"`
	ModelVersion  string               `json:"modelVersion"`
}

type googleCandidate struct {
	Content      *googleContentOut `json:"content,omitempty"`
	FinishReason string            `json:"finishReason,omitempty"`
	Index        int               `json:"index"`
}

type googleContentOut struct {
	Parts []GooglePart `json:"parts"`
	Role  string       `json:"role"`
}

type googleUsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// googleModelsResponse is the Google GET /v1beta/models payload.
type googleModelsResponse struct {
	Models []googleModelObject `json:"models"`
}

type googleModelObject struct {
	Name                       string   `json:"name"`
	DisplayName                string   `json:"displayName"`
	Description                string   `json:"description"`
	SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
}
