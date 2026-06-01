package apiconv

import (
	"encoding/base64"
	"encoding/json"
	"testing"
)

func TestDecodeDataURI(t *testing.T) {
	raw := base64.StdEncoding.EncodeToString([]byte("hello"))
	data, mime, ok := decodeDataURI("data:image/jpeg;base64," + raw)
	if !ok || string(data) != "hello" || mime != "image/jpeg" {
		t.Errorf("decodeDataURI = (%q, %q, %v)", data, mime, ok)
	}

	// Missing comma → not ok.
	if _, _, ok := decodeDataURI("data:image/png;base64"); ok {
		t.Errorf("expected failure without comma")
	}

	// Invalid base64 → not ok.
	if _, _, ok := decodeDataURI("data:image/png;base64,@@@notbase64@@@"); ok {
		t.Errorf("expected failure on bad base64")
	}
}

func TestImageFromURLString(t *testing.T) {
	// Remote URL → URL set, no data.
	img, ok := imageFromURLString("https://example.com/cat.jpg")
	if !ok || img.URL != "https://example.com/cat.jpg" || len(img.Data) != 0 {
		t.Errorf("remote: %+v ok=%v", img, ok)
	}
	if img.Filename != "cat.jpg" {
		t.Errorf("filename = %q, want cat.jpg", img.Filename)
	}

	// data URI → inline data, no URL.
	raw := base64.StdEncoding.EncodeToString([]byte("xy"))
	img, ok = imageFromURLString("data:image/png;base64," + raw)
	if !ok || string(img.Data) != "xy" || img.URL != "" {
		t.Errorf("data uri: %+v ok=%v", img, ok)
	}

	// Unsupported scheme → not ok.
	if _, ok := imageFromURLString("ftp://x/y"); ok {
		t.Errorf("ftp should be rejected")
	}
}

func TestImagesFromOpenAIMessages(t *testing.T) {
	raw := base64.StdEncoding.EncodeToString([]byte("img"))
	content := `[
		{"type":"text","text":"what is this?"},
		{"type":"image_url","image_url":{"url":"data:image/png;base64,` + raw + `"}}
	]`
	msgs := []Message{{Role: "user", Content: json.RawMessage(content)}}
	imgs := imagesFromOpenAIMessages(msgs)
	if len(imgs) != 1 || string(imgs[0].Data) != "img" {
		t.Fatalf("imgs = %+v", imgs)
	}
}

func TestImageFromPartBareStringAndInputImage(t *testing.T) {
	// Responses-style input_image with a bare string URL.
	img, ok := imageFromPart(map[string]any{"type": "input_image", "image_url": "https://e.com/a.png"})
	if !ok || img.URL != "https://e.com/a.png" {
		t.Errorf("input_image bare string: %+v ok=%v", img, ok)
	}
	// Non-image part → not ok.
	if _, ok := imageFromPart(map[string]any{"type": "text", "text": "hi"}); ok {
		t.Errorf("text part should not yield an image")
	}
}

func TestImagesFromAnthropicContent(t *testing.T) {
	raw := base64.StdEncoding.EncodeToString([]byte("ant"))
	content := `[
		{"type":"text","text":"caption"},
		{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"` + raw + `"}}
	]`
	imgs := imagesFromAnthropicContent(json.RawMessage(content))
	if len(imgs) != 1 || string(imgs[0].Data) != "ant" || imgs[0].Filename != "image.jpg" {
		t.Fatalf("imgs = %+v", imgs)
	}

	// URL source.
	imgs = imagesFromAnthropicContent(json.RawMessage(
		`[{"type":"image","source":{"type":"url","url":"https://e.com/p.webp"}}]`))
	if len(imgs) != 1 || imgs[0].URL != "https://e.com/p.webp" {
		t.Fatalf("url-source imgs = %+v", imgs)
	}
}

func TestImagesFromGoogleParts(t *testing.T) {
	raw := base64.StdEncoding.EncodeToString([]byte("ggl"))
	parts := []GooglePart{
		{Text: "hi"},
		{InlineData: &googleInlineData{MimeType: "image/webp", Data: raw}},
	}
	imgs := imagesFromGoogleParts(parts)
	if len(imgs) != 1 || string(imgs[0].Data) != "ggl" || imgs[0].Filename != "image.webp" {
		t.Fatalf("imgs = %+v", imgs)
	}
}

func TestImagesFromGoogleRequestJSON(t *testing.T) {
	raw := base64.StdEncoding.EncodeToString([]byte("Z"))
	body := `{"contents":[{"role":"user","parts":[
		{"text":"describe"},
		{"inlineData":{"mimeType":"image/png","data":"` + raw + `"}}
	]}]}`
	var req GenerateContentRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	imgs := imagesFromGoogleRequest(&req)
	if len(imgs) != 1 || string(imgs[0].Data) != "Z" {
		t.Fatalf("imgs = %+v", imgs)
	}
}

func TestFilenameForMime(t *testing.T) {
	cases := map[string]string{
		"image/png":  "image.png",
		"image/jpeg": "image.jpg",
		"image/webp": "image.webp",
		"":           "image.png",
		"weird":      "image.png",
	}
	for mime, want := range cases {
		if got := filenameForMime(mime); got != want {
			t.Errorf("filenameForMime(%q) = %q, want %q", mime, got, want)
		}
	}
}
