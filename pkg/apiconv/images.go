package apiconv

import (
	"encoding/base64"
	"encoding/json"
	"net/url"
	"path"
	"strings"

	"github.com/n0madic/go-gemini-web2api/pkg/gemini"
)

// ─── Extraction: OpenAI Chat Completions ─────────────────────────────────────

// imagesFromOpenAIMessages collects images from chat messages' content arrays.
func imagesFromOpenAIMessages(messages []Message) []gemini.InputImage {
	var out []gemini.InputImage
	for _, m := range messages {
		out = append(out, imagesFromRawContent(m.Content)...)
	}
	return out
}

// imagesFromRawContent extracts images from a polymorphic content value (a plain
// string has none; an array may contain image_url/input_image parts).
func imagesFromRawContent(raw json.RawMessage) []gemini.InputImage {
	if len(raw) == 0 {
		return nil
	}
	var parts []map[string]any
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil
	}
	var out []gemini.InputImage
	for _, p := range parts {
		if img, ok := imageFromPart(p); ok {
			out = append(out, img)
		}
	}
	return out
}

// imageFromPart extracts an InputImage from a generic content-part map, handling
// the OpenAI Chat ("image_url") and Responses ("input_image") shapes. The image
// reference may be an object {"url":…} or a bare string.
func imageFromPart(p map[string]any) (gemini.InputImage, bool) {
	switch typ, _ := p["type"].(string); typ {
	case "image_url", "input_image":
		switch v := p["image_url"].(type) {
		case string:
			return imageFromURLString(v)
		case map[string]any:
			if u, ok := v["url"].(string); ok {
				return imageFromURLString(u)
			}
		}
	}
	return gemini.InputImage{}, false
}

// ─── Extraction: Anthropic Messages ──────────────────────────────────────────

// imagesFromAnthropicRequest collects images from an Anthropic request's messages.
func imagesFromAnthropicRequest(req *AnthropicRequest) []gemini.InputImage {
	var out []gemini.InputImage
	for _, m := range req.Messages {
		out = append(out, imagesFromAnthropicContent(m.Content)...)
	}
	return out
}

// imagesFromAnthropicContent extracts images from an Anthropic content block array
// (image blocks with a base64 or url source).
func imagesFromAnthropicContent(raw json.RawMessage) []gemini.InputImage {
	if len(raw) == 0 {
		return nil
	}
	var blocks []map[string]any
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil
	}
	var out []gemini.InputImage
	for _, b := range blocks {
		if t, _ := b["type"].(string); t != "image" {
			continue
		}
		src, ok := b["source"].(map[string]any)
		if !ok {
			continue
		}
		switch src["type"] {
		case "base64":
			data, _ := src["data"].(string)
			mime, _ := src["media_type"].(string)
			if b, err := base64.StdEncoding.DecodeString(data); err == nil && len(b) > 0 {
				out = append(out, gemini.InputImage{Data: b, Filename: filenameForMime(mime)})
			}
		case "url":
			if u, ok := src["url"].(string); ok {
				if img, ok := imageFromURLString(u); ok {
					out = append(out, img)
				}
			}
		}
	}
	return out
}

// ─── Extraction: Google native ───────────────────────────────────────────────

// imagesFromGoogleRequest collects inline images from a Google generateContent
// request (system instruction and all content parts).
func imagesFromGoogleRequest(req *GenerateContentRequest) []gemini.InputImage {
	var out []gemini.InputImage
	if req.SystemInstruction != nil {
		out = append(out, imagesFromGoogleParts(req.SystemInstruction.Parts)...)
	}
	for _, content := range req.Contents {
		out = append(out, imagesFromGoogleParts(content.Parts)...)
	}
	return out
}

// imagesFromGoogleParts decodes inlineData (base64) image parts.
func imagesFromGoogleParts(parts []GooglePart) []gemini.InputImage {
	var out []gemini.InputImage
	for _, p := range parts {
		if p.InlineData == nil || p.InlineData.Data == "" {
			continue
		}
		b, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
		if err != nil || len(b) == 0 {
			continue
		}
		out = append(out, gemini.InputImage{Data: b, Filename: filenameForMime(p.InlineData.MimeType)})
	}
	return out
}

// ─── Extraction: OpenAI Responses API ────────────────────────────────────────

// imagesFromResponsesInput collects input_image parts from Responses API input items.
func imagesFromResponsesInput(req *ResponsesRequest) []gemini.InputImage {
	if len(req.Input) == 0 {
		return nil
	}
	var s string
	if json.Unmarshal(req.Input, &s) == nil {
		return nil // bare string carries no images
	}
	var items []json.RawMessage
	if json.Unmarshal(req.Input, &items) != nil {
		return nil
	}
	var out []gemini.InputImage
	for _, raw := range items {
		var item map[string]any
		if json.Unmarshal(raw, &item) != nil {
			continue
		}
		parts, ok := item["content"].([]any)
		if !ok {
			continue
		}
		for _, pp := range parts {
			if pm, ok := pp.(map[string]any); ok {
				if img, ok := imageFromPart(pm); ok {
					out = append(out, img)
				}
			}
		}
	}
	return out
}

// ─── Shared helpers ──────────────────────────────────────────────────────────

// imageFromURLString builds an InputImage from a data: URI (decoded inline) or a
// remote http(s) URL (fetched at upload time). Other schemes are rejected.
func imageFromURLString(s string) (gemini.InputImage, bool) {
	s = strings.TrimSpace(s)
	switch {
	case s == "":
		return gemini.InputImage{}, false
	case strings.HasPrefix(s, "data:"):
		data, mime, ok := decodeDataURI(s)
		if !ok {
			return gemini.InputImage{}, false
		}
		return gemini.InputImage{Data: data, Filename: filenameForMime(mime)}, true
	case strings.HasPrefix(s, "http://"), strings.HasPrefix(s, "https://"):
		return gemini.InputImage{URL: s, Filename: filenameForURL(s)}, true
	}
	return gemini.InputImage{}, false
}

// decodeDataURI decodes a "data:[<mime>][;base64],<payload>" URI into bytes + mime.
func decodeDataURI(s string) (data []byte, mime string, ok bool) {
	s = strings.TrimPrefix(s, "data:")
	comma := strings.IndexByte(s, ',')
	if comma < 0 {
		return nil, "", false
	}
	meta, payload := s[:comma], s[comma+1:]
	mime = "image/png"
	isBase64 := false
	for _, seg := range strings.Split(meta, ";") {
		switch {
		case seg == "base64":
			isBase64 = true
		case strings.Contains(seg, "/"):
			mime = seg
		}
	}
	if isBase64 {
		b, err := base64.StdEncoding.DecodeString(payload)
		if err != nil {
			return nil, "", false
		}
		return b, mime, true
	}
	dec, err := url.QueryUnescape(payload)
	if err != nil {
		return nil, "", false
	}
	return []byte(dec), mime, true
}

// filenameForMime maps an image MIME type to a representative filename.
func filenameForMime(mime string) string {
	switch strings.ToLower(strings.TrimSpace(mime)) {
	case "image/jpeg", "image/jpg":
		return "image.jpg"
	case "image/webp":
		return "image.webp"
	case "image/gif":
		return "image.gif"
	case "image/heic":
		return "image.heic"
	case "image/heif":
		return "image.heif"
	default:
		return "image.png"
	}
}

// filenameForURL derives a filename from a URL path, falling back to image.png.
func filenameForURL(u string) string {
	if parsed, err := url.Parse(u); err == nil {
		if base := path.Base(parsed.Path); strings.Contains(base, ".") {
			return base
		}
	}
	return "image.png"
}
