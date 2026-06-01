package apiconv

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestGoogleContentsToPrompt(t *testing.T) {
	req := &GenerateContentRequest{
		SystemInstruction: &GoogleContent{Parts: []GooglePart{{Text: "sys"}}},
		Contents: []GoogleContent{
			{Role: "user", Parts: []GooglePart{{Text: "hi"}}},
			{Role: "model", Parts: []GooglePart{{Text: "yo"}}},
		},
	}
	got := googleContentsToPrompt(req)
	for _, want := range []string{"[System instruction]: sys", "hi", "[Assistant]: yo"} {
		if !strings.Contains(got, want) {
			t.Errorf("prompt missing %q\nfull:\n%s", want, got)
		}
	}
}

func TestGoogleToolPrompt(t *testing.T) {
	req := &GenerateContentRequest{
		Tools: []GoogleTool{{FunctionDeclarations: []GoogleFunctionDecl{
			{Name: "get_weather", Description: "Get weather", Parameters: json.RawMessage(`{"type":"object"}`)},
		}}},
		Contents: []GoogleContent{{Role: "user", Parts: []GooglePart{{Text: "weather?"}}}},
	}
	if !req.hasTools() {
		t.Fatal("hasTools should be true")
	}
	prompt := googleContentsToPrompt(req)
	for _, want := range []string{"# Tool Use", "function_call", "get_weather"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt missing %q\n%s", want, prompt)
		}
	}
}

func TestGoogleToolChoiceAny(t *testing.T) {
	req := &GenerateContentRequest{
		Tools:      []GoogleTool{{FunctionDeclarations: []GoogleFunctionDecl{{Name: "f"}}}},
		ToolConfig: &GoogleToolConfig{FunctionCallingConfig: &GoogleFCConfig{Mode: "ANY"}},
		Contents:   []GoogleContent{{Role: "user", Parts: []GooglePart{{Text: "go"}}}},
	}
	prompt := googleContentsToPrompt(req)
	if !strings.Contains(prompt, "MUST call at least one tool") {
		t.Errorf("ANY constraint missing:\n%s", prompt)
	}
}

func TestGoogleToolChoiceAnyAllowed(t *testing.T) {
	req := &GenerateContentRequest{
		Tools: []GoogleTool{{FunctionDeclarations: []GoogleFunctionDecl{{Name: "f"}, {Name: "g"}}}},
		ToolConfig: &GoogleToolConfig{FunctionCallingConfig: &GoogleFCConfig{
			Mode: "ANY", AllowedFunctionNames: []string{"f", "g"}}},
		Contents: []GoogleContent{{Role: "user", Parts: []GooglePart{{Text: "go"}}}},
	}
	prompt := googleContentsToPrompt(req)
	if !strings.Contains(prompt, `MUST call one of these tools: "f", "g"`) {
		t.Errorf("allowed-names constraint missing:\n%s", prompt)
	}
}

func TestGoogleToolConfigNone(t *testing.T) {
	req := &GenerateContentRequest{
		Tools:      []GoogleTool{{FunctionDeclarations: []GoogleFunctionDecl{{Name: "f"}}}},
		ToolConfig: &GoogleToolConfig{FunctionCallingConfig: &GoogleFCConfig{Mode: "NONE"}},
		Contents:   []GoogleContent{{Role: "user", Parts: []GooglePart{{Text: "hi"}}}},
	}
	if req.hasTools() {
		t.Fatal("hasTools should be false when mode is NONE")
	}
	prompt := googleContentsToPrompt(req)
	if strings.Contains(prompt, "# Tool Use") {
		t.Errorf("NONE mode should omit the tools block:\n%s", prompt)
	}
}

func TestGoogleFunctionCallAndResponseInContent(t *testing.T) {
	req := &GenerateContentRequest{
		Contents: []GoogleContent{
			{Role: "user", Parts: []GooglePart{{Text: "do it"}}},
			{Role: "model", Parts: []GooglePart{{FunctionCall: &googleFunctionCall{
				Name: "search", Args: json.RawMessage(`{"q":"x"}`)}}}},
			{Role: "user", Parts: []GooglePart{{FunctionResponse: &googleFunctionResponse{
				Name: "search", Response: json.RawMessage(`{"hits":3}`)}}}},
		},
	}
	prompt := googleContentsToPrompt(req)
	for _, want := range []string{"```function_call", `"name": "search"`, "[Tool result for search]:", `{"hits":3}`} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt missing %q\n%s", want, prompt)
		}
	}
}

func TestParseGoogleFunctionCalls(t *testing.T) {
	t.Run("fenced", func(t *testing.T) {
		clean, calls := ParseGoogleFunctionCalls(
			"before\n```function_call\n{\"name\":\"get_x\",\"args\":{\"a\":1}}\n```\nafter")
		if len(calls) != 1 || calls[0].Name != "get_x" {
			t.Fatalf("calls = %+v", calls)
		}
		if !strings.Contains(string(calls[0].Args), `"a":1`) {
			t.Errorf("args = %s", calls[0].Args)
		}
		if !strings.Contains(clean, "before") || !strings.Contains(clean, "after") || strings.Contains(clean, "function_call") {
			t.Errorf("clean = %q", clean)
		}
	})

	t.Run("bare", func(t *testing.T) {
		_, calls := ParseGoogleFunctionCalls("function_call\n{\"name\":\"f\",\"args\":{}}")
		if len(calls) != 1 || calls[0].Name != "f" {
			t.Fatalf("calls = %+v", calls)
		}
	})

	t.Run("raw json", func(t *testing.T) {
		clean, calls := ParseGoogleFunctionCalls(`{"name":"g","args":{"q":"x"}}`)
		if len(calls) != 1 || calls[0].Name != "g" || clean != "" {
			t.Fatalf("calls = %+v clean = %q", calls, clean)
		}
	})

	t.Run("arguments fallback", func(t *testing.T) {
		_, calls := ParseGoogleFunctionCalls("```function_call\n{\"name\":\"h\",\"arguments\":{\"z\":2}}\n```")
		if len(calls) != 1 || !strings.Contains(string(calls[0].Args), `"z":2`) {
			t.Fatalf("calls = %+v", calls)
		}
	})

	t.Run("none", func(t *testing.T) {
		clean, calls := ParseGoogleFunctionCalls("just a normal answer")
		if len(calls) != 0 || clean != "just a normal answer" {
			t.Errorf("calls = %+v clean = %q", calls, clean)
		}
	})
}

func TestGoogleStreamError(t *testing.T) {
	// An interrupted Google stream must report a non-STOP finishReason so the
	// client can tell the generation did not complete normally.
	v := GoogleStreamError("gemini-3.5-flash", "the prompt", "partial output")
	resp, ok := v.(generateContentResponse)
	if !ok {
		t.Fatalf("type = %T, want generateContentResponse", v)
	}
	if len(resp.Candidates) != 1 {
		t.Fatalf("candidates = %d, want 1", len(resp.Candidates))
	}
	if resp.Candidates[0].FinishReason != "OTHER" {
		t.Errorf("finishReason = %q, want OTHER (not a clean STOP)", resp.Candidates[0].FinishReason)
	}
	if resp.UsageMetadata == nil || resp.UsageMetadata.TotalTokenCount == 0 {
		t.Errorf("usage metadata missing or empty: %+v", resp.UsageMetadata)
	}
}
