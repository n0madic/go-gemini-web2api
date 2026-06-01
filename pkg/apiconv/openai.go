// Package apiconv translates the OpenAI Chat Completions, OpenAI Responses,
// Anthropic Messages, and Google generateContent request/response formats to and
// from the prompt/text shape the gemini client speaks. It owns the request DTOs
// (with their Prompt builders) and the response/SSE-frame builders the HTTP layer
// serializes.
package apiconv

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/n0madic/go-gemini-web2api/pkg/gemini"
)

// toolCallRe matches a ```tool_call ... ``` block emitted by the model.
var toolCallRe = regexp.MustCompile("(?s)```tool_call\\s*\\n(.*?)\\n```")

// ─── Incoming request types (OpenAI Chat Completions) ────────────────────────

// ChatRequest is the OpenAI /v1/chat/completions request body.
type ChatRequest struct {
	Model      string          `json:"model"`
	Messages   []Message       `json:"messages"`
	Tools      []Tool          `json:"tools"`
	ToolChoice json.RawMessage `json:"tool_choice"`
	Stream     bool            `json:"stream"`
}

// Message is a single chat message. Content is polymorphic (string or an array
// of typed parts), so it is captured raw and decoded by contentText.
type Message struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	Name       string          `json:"name"`
	ToolCallID string          `json:"tool_call_id"` // correlates a tool-result message with its call
	ToolCalls  []ToolCall      `json:"tool_calls"`
}

// Tool is an OpenAI tool definition. It accepts both the wrapped
// ({"type":"function","function":{...}}) and flat ({"name":...}) shapes.
type Tool struct {
	Type     string        `json:"type"`
	Function *ToolFunction `json:"function"`
	// Flat fallback fields.
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ToolFunction is the inner function spec of a wrapped tool.
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ToolCall is an assistant tool invocation, used both inbound and outbound.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction holds the tool name and JSON-encoded arguments string.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Prompt builds the prompt, image attachments, and tool-active flag for an OpenAI
// chat request.
func (r *ChatRequest) Prompt() (prompt string, images []gemini.InputImage, toolsActive bool) {
	tc := parseOpenAIToolChoice(r.ToolChoice)
	toolsActive = len(r.Tools) > 0 && tc.mode != "none"
	prompt = messagesToPrompt(r.Messages, r.Tools, tc)
	images = imagesFromOpenAIMessages(r.Messages)
	return prompt, images, toolsActive
}

// contentText flattens a message's polymorphic content into a plain string.
func (m Message) contentText() string {
	return rawContentText(m.Content)
}

// rawContentText decodes a content value that may be a string or an array of
// {type, text} parts (text / input_text).
func rawContentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []map[string]any
	if err := json.Unmarshal(raw, &parts); err != nil {
		return ""
	}
	var sb strings.Builder
	for _, p := range parts {
		typ, _ := p["type"].(string)
		if typ == "text" || typ == "input_text" {
			if t, ok := p["text"].(string); ok {
				if sb.Len() > 0 {
					sb.WriteByte(' ')
				}
				sb.WriteString(t)
			}
		}
	}
	return sb.String()
}

// ─── Prompt construction ─────────────────────────────────────────────────────

// toolDef is the normalized tool shape embedded into the prompt.
type toolDef struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

// normalizeTools converts the inbound tools into a uniform list.
func normalizeTools(tools []Tool) []toolDef {
	out := make([]toolDef, 0, len(tools))
	for _, t := range tools {
		name, desc := t.Name, t.Description
		var params json.RawMessage = t.Parameters
		if t.Function != nil {
			if t.Function.Name != "" {
				name = t.Function.Name
			}
			if t.Function.Description != "" {
				desc = t.Function.Description
			}
			if len(t.Function.Parameters) > 0 {
				params = t.Function.Parameters
			}
		}
		var p any
		if len(params) > 0 {
			_ = json.Unmarshal(params, &p)
		}
		if p == nil {
			p = map[string]any{}
		}
		out = append(out, toolDef{Name: name, Description: desc, Parameters: p})
	}
	return out
}

// toolChoiceMode is a provider-agnostic representation of how/whether the model
// may call tools. mode is one of "", "auto", "none", "required", "function".
type toolChoiceMode struct {
	mode string
	name string // tool name, for mode == "function"
}

// parseOpenAIToolChoice maps the OpenAI tool_choice field onto toolChoiceMode.
func parseOpenAIToolChoice(raw json.RawMessage) toolChoiceMode {
	if len(raw) == 0 {
		return toolChoiceMode{}
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		switch s {
		case "none", "required", "auto":
			return toolChoiceMode{mode: s}
		}
		return toolChoiceMode{}
	}
	var obj struct {
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil && obj.Function.Name != "" {
		return toolChoiceMode{mode: "function", name: obj.Function.Name}
	}
	return toolChoiceMode{}
}

// parseAnthropicToolChoice maps the Anthropic tool_choice object onto toolChoiceMode.
func parseAnthropicToolChoice(raw json.RawMessage) toolChoiceMode {
	if len(raw) == 0 {
		return toolChoiceMode{}
	}
	var obj struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return toolChoiceMode{}
	}
	switch obj.Type {
	case "any":
		return toolChoiceMode{mode: "required"}
	case "tool":
		if obj.Name != "" {
			return toolChoiceMode{mode: "function", name: obj.Name}
		}
	case "none":
		return toolChoiceMode{mode: "none"}
	}
	return toolChoiceMode{}
}

// toolChoiceInstruction renders a constraint appended to the tool-use prompt.
func toolChoiceInstruction(tc toolChoiceMode) string {
	switch tc.mode {
	case "none":
		return "\n\nIMPORTANT: Do NOT call any tools. Respond with text only."
	case "required":
		return "\n\nIMPORTANT: You MUST call at least one tool. Do not respond with text only."
	case "function":
		return fmt.Sprintf("\n\nIMPORTANT: You MUST call the tool %q. Do not call other tools.", tc.name)
	}
	return ""
}

// messagesToPrompt converts OpenAI messages (plus optional tools and tool_choice)
// into a single prompt string with role markers.
func messagesToPrompt(messages []Message, tools []Tool, tc toolChoiceMode) string {
	var parts []string

	if len(tools) > 0 && tc.mode != "none" {
		defs := normalizeTools(tools)
		if len(defs) > 0 {
			defsJSON, _ := json.MarshalIndent(defs, "", "  ")
			parts = append(parts, "# Tool Use\n\n"+
				"You can call the following tools to help accomplish tasks. "+
				"These tools connect to the user's local environment and will execute when called.\n\n"+
				"Call format (use this exact format):\n"+
				"```tool_call\n{\"name\": \"func_name\", \"arguments\": {...}}\n```\n"+
				"When calling tools, output ONLY the tool_call block(s), nothing else.\n\n"+
				"Available tools:\n"+string(defsJSON)+
				toolChoiceInstruction(tc))
		}
	}

	for _, msg := range messages {
		content := msg.contentText()
		switch msg.Role {
		case "system":
			parts = append(parts, "[System instruction]: "+content)
		case "assistant":
			if len(msg.ToolCalls) > 0 {
				var tcStrs []string
				for _, tc := range msg.ToolCalls {
					args := tc.Function.Arguments
					if args == "" {
						args = "{}"
					}
					tcStrs = append(tcStrs, fmt.Sprintf(
						"```tool_call\n{\"name\": \"%s\", \"arguments\": %s}\n```",
						tc.Function.Name, args))
				}
				parts = append(parts, "[Assistant]: "+content+"\n"+strings.Join(tcStrs, "\n"))
			} else {
				parts = append(parts, "[Assistant]: "+content)
			}
		case "tool":
			// OpenAI tool-result messages carry tool_call_id (and sometimes name);
			// fall back to the id so the result is still labelled when name is absent.
			label := msg.Name
			if label == "" {
				label = msg.ToolCallID
			}
			parts = append(parts, "[Tool result for "+label+"]: "+content)
		default:
			if content != "" {
				parts = append(parts, content)
			}
		}
	}

	return joinNonEmpty(parts, "\n\n")
}

// ParseToolCalls extracts ```tool_call``` blocks, returning the cleaned text and
// the parsed tool calls.
func ParseToolCalls(text string) (string, []ToolCall) {
	var calls []ToolCall
	for _, m := range toolCallRe.FindAllStringSubmatch(text, -1) {
		var data struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(strings.TrimSpace(m[1])), &data); err != nil {
			continue
		}
		if data.Name == "" {
			continue
		}
		args := "{}"
		if len(data.Arguments) > 0 {
			args = string(data.Arguments)
		}
		calls = append(calls, ToolCall{
			ID:       "call_" + hexToken(8),
			Type:     "function",
			Function: ToolCallFunction{Name: data.Name, Arguments: args},
		})
	}
	clean := strings.TrimSpace(toolCallRe.ReplaceAllString(text, ""))
	return clean, calls
}

// joinNonEmpty joins only the non-empty parts with sep.
func joinNonEmpty(parts []string, sep string) string {
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p != "" {
			out = append(out, p)
		}
	}
	return strings.Join(out, sep)
}

// ─── Responses API (/v1/responses) ───────────────────────────────────────────

// ResponsesRequest is the OpenAI Responses API request body.
type ResponsesRequest struct {
	Model        string          `json:"model"`
	Instructions string          `json:"instructions"`
	Input        json.RawMessage `json:"input"`
	Tools        []Tool          `json:"tools"`
	ToolChoice   json.RawMessage `json:"tool_choice"`
	Stream       bool            `json:"stream"`
}

// Prompt builds the prompt, image attachments, and tool-active flag for an OpenAI
// Responses request.
func (r *ResponsesRequest) Prompt() (prompt string, images []gemini.InputImage, toolsActive bool) {
	tc := parseOpenAIToolChoice(r.ToolChoice)
	toolsActive = len(r.Tools) > 0 && tc.mode != "none"
	prompt = messagesToPrompt(responsesToMessages(r), r.Tools, tc)
	images = imagesFromResponsesInput(r)
	return prompt, images, toolsActive
}

// responsesToMessages flattens Responses API input items into chat messages.
func responsesToMessages(req *ResponsesRequest) []Message {
	var messages []Message
	if req.Instructions != "" {
		messages = append(messages, Message{Role: "system", Content: jsonString(req.Instructions)})
	}
	if len(req.Input) == 0 {
		return messages
	}

	// input may be a bare string.
	var asString string
	if err := json.Unmarshal(req.Input, &asString); err == nil {
		messages = append(messages, Message{Role: "user", Content: jsonString(asString)})
		return messages
	}

	// Otherwise it is an array of items (strings or objects).
	var items []json.RawMessage
	if err := json.Unmarshal(req.Input, &items); err != nil {
		return messages
	}
	for _, raw := range items {
		var s string
		if err := json.Unmarshal(raw, &s); err == nil {
			messages = append(messages, Message{Role: "user", Content: jsonString(s)})
			continue
		}
		var item map[string]any
		if err := json.Unmarshal(raw, &item); err != nil {
			continue
		}
		messages = append(messages, responsesItemToMessage(item))
	}
	return messages
}

// responsesItemToMessage converts a single Responses API input object.
func responsesItemToMessage(item map[string]any) Message {
	typ, _ := item["type"].(string)
	role, _ := item["role"].(string)

	switch {
	case typ == "function_call_output":
		callID, _ := item["call_id"].(string)
		name, _ := item["name"].(string)
		// Responses function_call_output usually omits the function name and only
		// carries call_id; keep the id so the tool result stays correlated.
		if name == "" {
			name = callID
		}
		output := stringifyContent(item["output"])
		return Message{Role: "tool", Name: name, ToolCallID: callID, Content: jsonString(output)}

	case role == "assistant":
		textAcc, toolCalls := extractAssistantContent(item["content"])
		return Message{Role: "assistant", Content: jsonString(textAcc), ToolCalls: toolCalls}

	default:
		if role == "" {
			role = "user"
		}
		content := stringifyContent(item["content"])
		return Message{Role: role, Content: jsonString(content)}
	}
}

// extractAssistantContent pulls output_text and function_call parts from a
// Responses assistant item's content array.
func extractAssistantContent(content any) (string, []ToolCall) {
	parts, ok := content.([]any)
	if !ok {
		if s, ok := content.(string); ok {
			return s, nil
		}
		return "", nil
	}
	var sb strings.Builder
	var calls []ToolCall
	for i, p := range parts {
		cp, ok := p.(map[string]any)
		if !ok {
			continue
		}
		switch cp["type"] {
		case "output_text":
			if t, ok := cp["text"].(string); ok {
				sb.WriteString(t)
			}
		case "function_call":
			name, _ := cp["name"].(string)
			args, _ := cp["arguments"].(string)
			if args == "" {
				args = "{}"
			}
			id, _ := cp["call_id"].(string)
			if id == "" {
				id = fmt.Sprintf("call_%d", i)
			}
			calls = append(calls, ToolCall{
				ID:       id,
				Type:     "function",
				Function: ToolCallFunction{Name: name, Arguments: args},
			})
		}
	}
	return sb.String(), calls
}

// stringifyContent renders a content value (string or array of text parts) to text.
func stringifyContent(v any) string {
	switch c := v.(type) {
	case string:
		return c
	case []any:
		var sb strings.Builder
		for _, p := range c {
			cp, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if typ, _ := cp["type"].(string); typ == "text" || typ == "input_text" {
				if t, ok := cp["text"].(string); ok {
					if sb.Len() > 0 {
						sb.WriteByte(' ')
					}
					sb.WriteString(t)
				}
			}
		}
		return sb.String()
	default:
		return ""
	}
}

// jsonString wraps a plain string as a json.RawMessage for use as Message.Content.
func jsonString(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}
