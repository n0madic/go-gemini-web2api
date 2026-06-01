package apiconv

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestMessagesToPrompt(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: jsonString("be concise")},
		{Role: "user", Content: jsonString("hello")},
		{Role: "assistant", Content: jsonString("hi there")},
		{Role: "tool", Name: "search", Content: jsonString("found it")},
	}
	got := messagesToPrompt(msgs, nil, toolChoiceMode{})

	for _, want := range []string{
		"[System instruction]: be concise",
		"hello",
		"[Assistant]: hi there",
		"[Tool result for search]: found it",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("prompt missing %q\nfull:\n%s", want, got)
		}
	}
}

func TestMessagesToPromptArrayContent(t *testing.T) {
	content := json.RawMessage(`[{"type":"text","text":"part1"},{"type":"input_text","text":"part2"}]`)
	msgs := []Message{{Role: "user", Content: content}}
	got := messagesToPrompt(msgs, nil, toolChoiceMode{})
	if got != "part1 part2" {
		t.Errorf("got %q, want %q", got, "part1 part2")
	}
}

func TestMessagesToPromptWithTools(t *testing.T) {
	tools := []Tool{{
		Type: "function",
		Function: &ToolFunction{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters:  json.RawMessage(`{"type":"object"}`),
		},
	}}
	msgs := []Message{{Role: "user", Content: jsonString("weather?")}}
	got := messagesToPrompt(msgs, tools, toolChoiceMode{})
	if !strings.Contains(got, "Available tools") || !strings.Contains(got, "get_weather") {
		t.Errorf("tools block missing:\n%s", got)
	}
}

func TestMessagesToPromptFlatTool(t *testing.T) {
	// Flat tool shape (no "function" wrapper).
	tools := []Tool{{Name: "do_thing", Description: "does", Parameters: json.RawMessage(`{}`)}}
	got := messagesToPrompt([]Message{{Role: "user", Content: jsonString("x")}}, tools, toolChoiceMode{})
	if !strings.Contains(got, "do_thing") {
		t.Errorf("flat tool name missing:\n%s", got)
	}
}

func TestParseToolCalls(t *testing.T) {
	text := "before text\n```tool_call\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}\n```\nafter text"
	clean, calls := ParseToolCalls(text)

	if len(calls) != 1 {
		t.Fatalf("calls = %d, want 1", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Errorf("name = %q, want get_weather", calls[0].Function.Name)
	}
	if !strings.Contains(calls[0].Function.Arguments, "NYC") {
		t.Errorf("arguments = %q, want NYC", calls[0].Function.Arguments)
	}
	if !strings.HasPrefix(calls[0].ID, "call_") {
		t.Errorf("id = %q, want call_ prefix", calls[0].ID)
	}
	if strings.Contains(clean, "tool_call") {
		t.Errorf("clean text still has tool_call block: %q", clean)
	}
	if !strings.Contains(clean, "before text") || !strings.Contains(clean, "after text") {
		t.Errorf("clean text lost surrounding content: %q", clean)
	}
}

func TestParseToolCallsNone(t *testing.T) {
	clean, calls := ParseToolCalls("just a normal reply")
	if len(calls) != 0 {
		t.Errorf("calls = %d, want 0", len(calls))
	}
	if clean != "just a normal reply" {
		t.Errorf("clean = %q", clean)
	}
}

func TestRawContentText(t *testing.T) {
	if got := rawContentText(jsonString("plain")); got != "plain" {
		t.Errorf("string content = %q, want plain", got)
	}
	if got := rawContentText(nil); got != "" {
		t.Errorf("nil content = %q, want empty", got)
	}
	arr := json.RawMessage(`[{"type":"text","text":"a"},{"type":"image","url":"x"},{"type":"text","text":"b"}]`)
	if got := rawContentText(arr); got != "a b" {
		t.Errorf("array content = %q, want %q", got, "a b")
	}
}

func TestResponsesToMessages(t *testing.T) {
	req := &ResponsesRequest{
		Instructions: "system prompt",
		Input:        json.RawMessage(`"hello there"`),
	}
	msgs := responsesToMessages(req)
	if len(msgs) != 2 {
		t.Fatalf("messages = %d, want 2", len(msgs))
	}
	if msgs[0].Role != "system" || msgs[0].contentText() != "system prompt" {
		t.Errorf("msg[0] = %+v", msgs[0])
	}
	if msgs[1].Role != "user" || msgs[1].contentText() != "hello there" {
		t.Errorf("msg[1] = %+v", msgs[1])
	}
}

func TestResponsesToMessagesItems(t *testing.T) {
	req := &ResponsesRequest{
		Input: json.RawMessage(`[
			{"role":"user","content":[{"type":"input_text","text":"q"}]},
			{"role":"assistant","content":[{"type":"output_text","text":"a"}]},
			{"type":"function_call_output","call_id":"c1","name":"fn","output":"42"}
		]`),
	}
	msgs := responsesToMessages(req)
	if len(msgs) != 3 {
		t.Fatalf("messages = %d, want 3", len(msgs))
	}
	if msgs[0].Role != "user" || msgs[0].contentText() != "q" {
		t.Errorf("msg[0] = %+v", msgs[0])
	}
	if msgs[1].Role != "assistant" || msgs[1].contentText() != "a" {
		t.Errorf("msg[1] = %+v", msgs[1])
	}
	if msgs[2].Role != "tool" || msgs[2].Name != "fn" || msgs[2].contentText() != "42" {
		t.Errorf("msg[2] = %+v", msgs[2])
	}
}

func TestApproxTokens(t *testing.T) {
	if got := ApproxTokens("abcdefgh"); got != 2 {
		t.Errorf("ApproxTokens(8 chars) = %d, want 2", got)
	}
}

func TestParseToolChoice(t *testing.T) {
	// OpenAI forms.
	if tc := parseOpenAIToolChoice(json.RawMessage(`"none"`)); tc.mode != "none" {
		t.Errorf("openai none = %+v", tc)
	}
	if tc := parseOpenAIToolChoice(json.RawMessage(`"required"`)); tc.mode != "required" {
		t.Errorf("openai required = %+v", tc)
	}
	if tc := parseOpenAIToolChoice(json.RawMessage(`{"type":"function","function":{"name":"f"}}`)); tc.mode != "function" || tc.name != "f" {
		t.Errorf("openai function = %+v", tc)
	}
	if tc := parseOpenAIToolChoice(nil); tc.mode != "" {
		t.Errorf("openai default = %+v", tc)
	}
	// Anthropic forms.
	if tc := parseAnthropicToolChoice(json.RawMessage(`{"type":"any"}`)); tc.mode != "required" {
		t.Errorf("anthropic any = %+v", tc)
	}
	if tc := parseAnthropicToolChoice(json.RawMessage(`{"type":"tool","name":"g"}`)); tc.mode != "function" || tc.name != "g" {
		t.Errorf("anthropic tool = %+v", tc)
	}
	if tc := parseAnthropicToolChoice(json.RawMessage(`{"type":"none"}`)); tc.mode != "none" {
		t.Errorf("anthropic none = %+v", tc)
	}
}

func TestMessagesToPromptToolChoice(t *testing.T) {
	tools := []Tool{{Name: "f", Description: "d", Parameters: json.RawMessage(`{}`)}}
	msgs := []Message{{Role: "user", Content: jsonString("x")}}

	if got := messagesToPrompt(msgs, tools, toolChoiceMode{mode: "none"}); strings.Contains(got, "Tool Use") {
		t.Errorf("none should omit the tools block:\n%s", got)
	}
	if got := messagesToPrompt(msgs, tools, toolChoiceMode{mode: "required"}); !strings.Contains(got, "MUST call at least one tool") {
		t.Errorf("required constraint missing:\n%s", got)
	}
	got := messagesToPrompt(msgs, tools, toolChoiceMode{mode: "function", name: "f"})
	if !strings.Contains(got, `MUST call the tool "f"`) {
		t.Errorf("function constraint missing:\n%s", got)
	}
}

func TestToolResultLabelFallsBackToID(t *testing.T) {
	// An OpenAI tool-result message often carries only tool_call_id (no name);
	// the id must still label the result in the prompt.
	body := `{"model":"x","messages":[
		{"role":"user","content":"hi"},
		{"role":"tool","tool_call_id":"call_abc","content":"42"}
	]}`
	var req ChatRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	prompt, _, _ := req.Prompt()
	if !strings.Contains(prompt, "[Tool result for call_abc]: 42") {
		t.Errorf("expected id-labelled tool result, got:\n%s", prompt)
	}
}

func TestResponsesFunctionOutputUsesCallID(t *testing.T) {
	// A Responses function_call_output usually omits the function name and only
	// carries call_id; the id must still label the tool result.
	body := `{"model":"x","input":[
		{"type":"function_call_output","call_id":"call_xyz","output":"done"}
	]}`
	var req ResponsesRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	prompt, _, _ := req.Prompt()
	if !strings.Contains(prompt, "[Tool result for call_xyz]: done") {
		t.Errorf("expected call_id-labelled tool result, got:\n%s", prompt)
	}
}
