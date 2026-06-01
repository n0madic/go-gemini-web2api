package apiconv

import (
	"encoding/json"
	"strings"

	"github.com/n0madic/go-gemini-web2api/pkg/gemini"
)

// modelCreatedAt is the static creation timestamp reported for Anthropic models.
const modelCreatedAt = "2025-01-01T00:00:00Z"

// ─── Incoming request types (Anthropic Messages API) ─────────────────────────

// AnthropicRequest is the POST /v1/messages request body. Unsupported fields
// (temperature, top_p, stop_sequences, metadata, ...) are accepted and ignored,
// since the Gemini web backend does not expose them.
type AnthropicRequest struct {
	Model      string             `json:"model"`
	MaxTokens  int                `json:"max_tokens"`
	System     json.RawMessage    `json:"system"` // string or []text-block
	Messages   []AnthropicMessage `json:"messages"`
	Tools      []AnthropicTool    `json:"tools"`
	ToolChoice json.RawMessage    `json:"tool_choice"`
	Stream     bool               `json:"stream"`
}

// AnthropicMessage is one message; Content is polymorphic (string or block array).
type AnthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// AnthropicTool is a tool definition. input_schema maps onto our generic parameters.
type AnthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// Prompt builds the prompt, image attachments, and tool-active flag for an
// Anthropic messages request.
func (r *AnthropicRequest) Prompt() (prompt string, images []gemini.InputImage, toolsActive bool) {
	tc := parseAnthropicToolChoice(r.ToolChoice)
	tools := anthropicToolsToTools(r.Tools)
	toolsActive = len(tools) > 0 && tc.mode != "none"
	prompt = messagesToPrompt(anthropicToMessages(r), tools, tc)
	images = imagesFromAnthropicRequest(r)
	return prompt, images, toolsActive
}

// ─── Request conversion ──────────────────────────────────────────────────────

// anthropicToMessages flattens an Anthropic request into our internal message
// list so the shared messagesToPrompt builder can be reused.
func anthropicToMessages(req *AnthropicRequest) []Message {
	var msgs []Message
	if sys := rawContentText(req.System); sys != "" {
		msgs = append(msgs, Message{Role: "system", Content: jsonString(sys)})
	}
	for _, m := range req.Messages {
		msgs = append(msgs, anthropicMessageToMessages(m)...)
	}
	return msgs
}

// anthropicMessageToMessages converts a single Anthropic message. A message may
// expand into several internal messages: tool_result blocks become separate
// "tool" messages preceding the text/tool_use content.
func anthropicMessageToMessages(m AnthropicMessage) []Message {
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return []Message{{Role: m.Role, Content: jsonString(s)}}
	}

	var blocks []map[string]any
	if err := json.Unmarshal(m.Content, &blocks); err != nil {
		return nil
	}

	var text strings.Builder
	var toolCalls []ToolCall
	var out []Message
	for _, b := range blocks {
		switch b["type"] {
		case "text":
			if t, ok := b["text"].(string); ok {
				text.WriteString(t)
			}
		case "tool_use":
			id, _ := b["id"].(string)
			name, _ := b["name"].(string)
			args := "{}"
			if raw, err := json.Marshal(b["input"]); err == nil && b["input"] != nil {
				args = string(raw)
			}
			toolCalls = append(toolCalls, ToolCall{
				ID: id, Type: "function",
				Function: ToolCallFunction{Name: name, Arguments: args},
			})
		case "tool_result":
			tid, _ := b["tool_use_id"].(string)
			out = append(out, Message{
				Role: "tool", Name: tid,
				Content: jsonString(stringifyContent(b["content"])),
			})
		}
	}
	out = append(out, Message{Role: m.Role, Content: jsonString(text.String()), ToolCalls: toolCalls})
	return out
}

// anthropicToolsToTools maps Anthropic tools onto our generic tool type.
func anthropicToolsToTools(tools []AnthropicTool) []Tool {
	out := make([]Tool, 0, len(tools))
	for _, t := range tools {
		out = append(out, Tool{Name: t.Name, Description: t.Description, Parameters: t.InputSchema})
	}
	return out
}

// ─── Response builders ───────────────────────────────────────────────────────

// AnthropicResponse builds a complete (non-streaming) message response.
func AnthropicResponse(model, prompt, text string, toolCalls []ToolCall) any {
	return buildAnthropicResponse(model, prompt, text, toolCalls)
}

// buildAnthropicResponse builds the concrete (non-streaming) message response. It
// returns the unexported DTO so the package's own tests and the streaming
// builders can read it; the exported AnthropicResponse hands it out as any.
func buildAnthropicResponse(model, prompt, text string, toolCalls []ToolCall) anthropicResponse {
	var content []anthropicContentBlock
	if text != "" {
		content = append(content, anthropicContentBlock{Type: "text", Text: text})
	}
	for _, tc := range toolCalls {
		input := json.RawMessage(tc.Function.Arguments)
		if len(input) == 0 {
			input = json.RawMessage("{}")
		}
		content = append(content, anthropicContentBlock{
			Type: "tool_use", ID: "toolu_" + hexToken(24), Name: tc.Function.Name, Input: input,
		})
	}
	if len(content) == 0 {
		content = append(content, anthropicContentBlock{Type: "text", Text: ""})
	}

	stop := "end_turn"
	if len(toolCalls) > 0 {
		stop = "tool_use"
	}
	return anthropicResponse{
		ID: "msg_" + hexToken(24), Type: "message", Role: "assistant", Model: model,
		Content: content, StopReason: contentPtr(stop),
		Usage: anthropicUsage{InputTokens: ApproxTokens(prompt), OutputTokens: ApproxTokens(text)},
	}
}

// AnthropicCountTokens builds the POST /v1/messages/count_tokens response.
func AnthropicCountTokens(prompt string) any {
	return anthropicCountTokensResponse{InputTokens: ApproxTokens(prompt)}
}

// AnthropicError builds an Anthropic-shaped error envelope.
func AnthropicError(msg string) any {
	return anthropicErrorResponse{
		Type:  "error",
		Error: anthropicErrorDetail{Type: "invalid_request_error", Message: msg},
	}
}

// AnthropicModelList builds the Anthropic GET /v1/models payload.
func AnthropicModelList(models []*gemini.AvailableModel) any {
	data := make([]anthropicModelObject, 0, len(models))
	for _, m := range models {
		data = append(data, anthropicModelObject{
			Type: "model", ID: m.Name, DisplayName: m.Name, CreatedAt: modelCreatedAt,
		})
	}
	resp := anthropicModelsResponse{Data: data, HasMore: false}
	if len(data) > 0 {
		resp.FirstID = data[0].ID
		resp.LastID = data[len(data)-1].ID
	}
	return resp
}

// ─── Streaming event builders ────────────────────────────────────────────────
//
// The HTTP layer owns the SSE event names and the emit loop; these builders own
// each event's JSON payload.

// AnthropicMessageStart builds the message_start event payload.
func AnthropicMessageStart(id, model string, inputTokens int) any {
	return anthropicStreamStart{
		Type: "message_start",
		Message: anthropicResponse{
			ID: id, Type: "message", Role: "assistant", Model: model,
			Content: []anthropicContentBlock{},
			Usage:   anthropicUsage{InputTokens: inputTokens, OutputTokens: 0},
		},
	}
}

// AnthropicTextBlockStart builds a content_block_start payload for a text block.
func AnthropicTextBlockStart(index int) any {
	return anthropicBlockStart{
		Type: "content_block_start", Index: index,
		ContentBlock: anthropicTextBlock{Type: "text", Text: ""},
	}
}

// AnthropicTextDelta builds a content_block_delta payload carrying a text delta.
func AnthropicTextDelta(index int, text string) any {
	return anthropicBlockDelta{
		Type: "content_block_delta", Index: index,
		Delta: anthropicTextDelta{Type: "text_delta", Text: text},
	}
}

// AnthropicToolUseBlockStart builds a content_block_start payload for a tool_use block.
func AnthropicToolUseBlockStart(index int, tc ToolCall) any {
	return anthropicBlockStart{
		Type: "content_block_start", Index: index,
		ContentBlock: anthropicToolUseStart{
			Type: "tool_use", ID: "toolu_" + hexToken(24), Name: tc.Function.Name, Input: json.RawMessage("{}"),
		},
	}
}

// AnthropicToolUseDelta builds a content_block_delta payload carrying the tool
// call's arguments as an input_json_delta.
func AnthropicToolUseDelta(index int, tc ToolCall) any {
	args := tc.Function.Arguments
	if args == "" {
		args = "{}"
	}
	return anthropicBlockDelta{
		Type: "content_block_delta", Index: index,
		Delta: anthropicInputJSONDelta{Type: "input_json_delta", PartialJSON: args},
	}
}

// AnthropicBlockStop builds a content_block_stop payload.
func AnthropicBlockStop(index int) any {
	return anthropicBlockStop{Type: "content_block_stop", Index: index}
}

// AnthropicMessageDelta builds the trailing message_delta payload (stop reason +
// output token count).
func AnthropicMessageDelta(stop string, outputTokens int) any {
	return anthropicMessageDeltaEvent{
		Type:  "message_delta",
		Delta: anthropicMessageDelta{StopReason: contentPtr(stop)},
		Usage: anthropicUsageDelta{OutputTokens: outputTokens},
	}
}

// AnthropicMessageStop builds the message_stop payload.
func AnthropicMessageStop() any {
	return anthropicMessageStop{Type: "message_stop"}
}

// ─── Output types ────────────────────────────────────────────────────────────

type anthropicResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Model        string                  `json:"model"`
	Content      []anthropicContentBlock `json:"content"`
	StopReason   *string                 `json:"stop_reason"`
	StopSequence *string                 `json:"stop_sequence"`
	Usage        anthropicUsage          `json:"usage"`
}

type anthropicContentBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthropicCountTokensResponse struct {
	InputTokens int `json:"input_tokens"`
}

type anthropicModelsResponse struct {
	Data    []anthropicModelObject `json:"data"`
	HasMore bool                   `json:"has_more"`
	FirstID string                 `json:"first_id"`
	LastID  string                 `json:"last_id"`
}

type anthropicModelObject struct {
	Type        string `json:"type"`
	ID          string `json:"id"`
	DisplayName string `json:"display_name"`
	CreatedAt   string `json:"created_at"`
}

type anthropicErrorResponse struct {
	Type  string               `json:"type"`
	Error anthropicErrorDetail `json:"error"`
}

type anthropicErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// ─── Streaming event types ───────────────────────────────────────────────────

type anthropicStreamStart struct {
	Type    string            `json:"type"`
	Message anthropicResponse `json:"message"`
}

type anthropicBlockStart struct {
	Type         string `json:"type"`
	Index        int    `json:"index"`
	ContentBlock any    `json:"content_block"`
}

type anthropicTextBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anthropicToolUseStart struct {
	Type  string          `json:"type"`
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

type anthropicBlockDelta struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta any    `json:"delta"`
}

type anthropicTextDelta struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anthropicInputJSONDelta struct {
	Type        string `json:"type"`
	PartialJSON string `json:"partial_json"`
}

type anthropicBlockStop struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

type anthropicMessageDeltaEvent struct {
	Type  string                `json:"type"`
	Delta anthropicMessageDelta `json:"delta"`
	Usage anthropicUsageDelta   `json:"usage"`
}

type anthropicMessageDelta struct {
	StopReason   *string `json:"stop_reason"`
	StopSequence *string `json:"stop_sequence"`
}

type anthropicUsageDelta struct {
	OutputTokens int `json:"output_tokens"`
}

type anthropicMessageStop struct {
	Type string `json:"type"`
}
