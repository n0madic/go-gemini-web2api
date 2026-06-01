package apiconv

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestAnthropicToMessages(t *testing.T) {
	req := &AnthropicRequest{
		System: json.RawMessage(`"be brief"`),
		Messages: []AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`[{"type":"text","text":"hello"}]`)},
			{Role: "assistant", Content: json.RawMessage(
				`[{"type":"text","text":"hi"},{"type":"tool_use","id":"toolu_1","name":"get_x","input":{"a":1}}]`)},
			{Role: "user", Content: json.RawMessage(
				`[{"type":"tool_result","tool_use_id":"toolu_1","content":"42"}]`)},
		},
	}
	prompt := messagesToPrompt(anthropicToMessages(req), nil, toolChoiceMode{})

	for _, want := range []string{
		"[System instruction]: be brief",
		"hello",
		"[Assistant]: hi",
		"get_x",
		"[Tool result for toolu_1]: 42",
	} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt missing %q\nfull:\n%s", want, prompt)
		}
	}
}

func TestAnthropicToMessagesStringContent(t *testing.T) {
	req := &AnthropicRequest{
		Messages: []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"plain string"`)}},
	}
	msgs := anthropicToMessages(req)
	if len(msgs) != 1 || msgs[0].Role != "user" || msgs[0].contentText() != "plain string" {
		t.Errorf("got %+v", msgs)
	}
}

func TestAnthropicToolsToTools(t *testing.T) {
	tools := []AnthropicTool{{
		Name:        "get_weather",
		Description: "Get weather",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
	}}
	got := anthropicToolsToTools(tools)
	if len(got) != 1 || got[0].Name != "get_weather" {
		t.Fatalf("got %+v", got)
	}
	prompt := messagesToPrompt([]Message{{Role: "user", Content: jsonString("x")}}, got, toolChoiceMode{})
	if !strings.Contains(prompt, "get_weather") || !strings.Contains(prompt, "Available tools") {
		t.Errorf("tools not rendered into prompt:\n%s", prompt)
	}
}

func TestAnthropicResponseBuilder(t *testing.T) {
	t.Run("text", func(t *testing.T) {
		resp := buildAnthropicResponse("gemini-3.5-flash", "prompt", "hello", nil)
		if resp.Type != "message" || resp.Role != "assistant" {
			t.Errorf("envelope = %+v", resp)
		}
		if len(resp.Content) != 1 || resp.Content[0].Type != "text" || resp.Content[0].Text != "hello" {
			t.Errorf("content = %+v", resp.Content)
		}
		if resp.StopReason == nil || *resp.StopReason != "end_turn" {
			t.Errorf("stop_reason = %v", resp.StopReason)
		}
	})

	t.Run("tool_use", func(t *testing.T) {
		calls := []ToolCall{{Function: ToolCallFunction{Name: "f", Arguments: `{"x":1}`}}}
		resp := buildAnthropicResponse("gemini-3.5-flash", "p", "", calls)
		if resp.StopReason == nil || *resp.StopReason != "tool_use" {
			t.Errorf("stop_reason = %v", resp.StopReason)
		}
		if len(resp.Content) != 1 || resp.Content[0].Type != "tool_use" || resp.Content[0].Name != "f" {
			t.Fatalf("content = %+v", resp.Content)
		}
		if !strings.HasPrefix(resp.Content[0].ID, "toolu_") {
			t.Errorf("tool id = %q, want toolu_ prefix", resp.Content[0].ID)
		}
	})
}
