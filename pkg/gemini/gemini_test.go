package gemini

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/url"
	"regexp"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

func testConfig() Config {
	return Config{
		RetryAttempts:  1,
		RetryDelaySec:  0,
		RequestTimeout: 10,
		GeminiBL:       "test_bl",
		DefaultModel:   "gemini-3.5-flash",
	}
}

// testLogger returns a logger that discards all output.
func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func TestNewAppliesDefaults(t *testing.T) {
	// An empty Config (and a nil logger) must yield a usable client with all
	// upstream fields defaulted.
	c, err := New(Config{}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if c.log == nil {
		t.Errorf("logger should default to non-nil when nil is passed")
	}
	if c.cfg.RequestTimeout != DefaultRequestTimeout {
		t.Errorf("RequestTimeout = %d, want %d", c.cfg.RequestTimeout, DefaultRequestTimeout)
	}
	if c.cfg.RetryAttempts != DefaultRetryAttempts {
		t.Errorf("RetryAttempts = %d, want %d", c.cfg.RetryAttempts, DefaultRetryAttempts)
	}
	if c.cfg.RetryDelaySec != DefaultRetryDelaySec {
		t.Errorf("RetryDelaySec = %d, want %d", c.cfg.RetryDelaySec, DefaultRetryDelaySec)
	}
	if c.cfg.DefaultModel != DefaultModel {
		t.Errorf("DefaultModel = %q, want %q", c.cfg.DefaultModel, DefaultModel)
	}
	if c.cfg.GeminiBL != DefaultBuildLabel || c.bl != DefaultBuildLabel {
		t.Errorf("GeminiBL = %q / bl = %q, want %q", c.cfg.GeminiBL, c.bl, DefaultBuildLabel)
	}
	// No explicit build label → auto-resolution is turned on by default.
	if !c.cfg.GeminiBLAuto || !c.blAuto {
		t.Errorf("GeminiBLAuto = %v / blAuto = %v, want true (auto-resolve when no explicit BL)", c.cfg.GeminiBLAuto, c.blAuto)
	}

	// An explicit build label pins it and leaves auto-resolution off.
	cPinned, err := New(Config{GeminiBL: "boq_pinned_123"}, testLogger())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if cPinned.cfg.GeminiBL != "boq_pinned_123" || cPinned.blAuto {
		t.Errorf("pinned BL: bl=%q auto=%v, want boq_pinned_123 / false", cPinned.cfg.GeminiBL, cPinned.blAuto)
	}
	if c.http.Timeout != time.Duration(DefaultRequestTimeout)*time.Second {
		t.Errorf("http timeout = %v, want %v", c.http.Timeout, time.Duration(DefaultRequestTimeout)*time.Second)
	}

	// Explicitly-set fields are preserved (not overwritten by defaults).
	c2, err := New(Config{RetryAttempts: 1, RequestTimeout: 5, DefaultModel: "gemini-3.1-pro"}, testLogger())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if c2.cfg.RetryAttempts != 1 || c2.cfg.RequestTimeout != 5 || c2.cfg.DefaultModel != "gemini-3.1-pro" {
		t.Errorf("explicit values overwritten: %+v", c2.cfg)
	}
}

func TestBuildBody(t *testing.T) {
	c, err := New(testConfig(), testLogger())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	c.setAt("tok123") // a scraped at token is echoed as the "at" form field

	body, reqUUID, err := c.buildBody(GenParams{Prompt: "hello prompt"}, nil)
	if err != nil {
		t.Fatalf("buildBody: %v", err)
	}

	// The returned UUID is an uppercase v4 UUID (it must echo inner[59]).
	uuidRe := regexp.MustCompile(`^[0-9A-F]{8}-[0-9A-F]{4}-4[0-9A-F]{3}-[89AB][0-9A-F]{3}-[0-9A-F]{12}$`)
	if !uuidRe.MatchString(reqUUID) {
		t.Errorf("reqUUID = %q, want uppercase UUID", reqUUID)
	}

	vals, err := url.ParseQuery(body)
	if err != nil {
		t.Fatalf("ParseQuery: %v", err)
	}
	if vals.Get("at") != "tok123" {
		t.Errorf("at = %q, want tok123", vals.Get("at"))
	}

	var outer []any
	if err := json.Unmarshal([]byte(vals.Get("f.req")), &outer); err != nil {
		t.Fatalf("unmarshal f.req: %v", err)
	}
	if len(outer) != 2 || outer[0] != nil {
		t.Fatalf("outer = %v, want [null, <inner>]", outer)
	}
	innerStr, ok := outer[1].(string)
	if !ok {
		t.Fatalf("outer[1] is not a string: %T", outer[1])
	}

	var inner []any
	if err := json.Unmarshal([]byte(innerStr), &inner); err != nil {
		t.Fatalf("unmarshal inner: %v", err)
	}
	// The new payload is a sparse 69-element array; there is no inner[79] (mode).
	if len(inner) != 69 {
		t.Fatalf("inner length = %d, want 69", len(inner))
	}

	// inner[0][0] is the prompt.
	if first, ok := inner[0].([]any); !ok || first[0] != "hello prompt" {
		t.Errorf("inner[0] = %v, want prompt at index 0", inner[0])
	}
	// inner[17] is the constant [[0]] (reasoning level is not exposed).
	if got, ok := inner[17].([]any); !ok || len(got) != 1 {
		t.Errorf("inner[17] = %v, want [[0]]", inner[17])
	} else if g2, ok := got[0].([]any); !ok || len(g2) != 1 || g2[0] != float64(0) {
		t.Errorf("inner[17][0] = %v, want [0]", got[0])
	}
	// inner[61] must serialize as an empty array, not null.
	if got, ok := inner[61].([]any); !ok || len(got) != 0 {
		t.Errorf("inner[61] = %v, want []", inner[61])
	}
	// inner[59] is the same UUID returned for the selection header.
	if s, ok := inner[59].(string); !ok || s != reqUUID {
		t.Errorf("inner[59] = %v, want %q", inner[59], reqUUID)
	}
}

func TestNewRequestModelHeaders(t *testing.T) {
	c, _ := New(testConfig(), testLogger())

	t.Run("with model id", func(t *testing.T) {
		m := &AvailableModel{Name: "gemini-3.1-pro", ModelID: "proid", Capacity: 2, CapacityField: 12}
		req, err := c.newRequest(context.Background(), "f.req=x", m, "REQ-UUID")
		if err != nil {
			t.Fatalf("newRequest: %v", err)
		}
		if req.Header.Get("x-goog-ext-525001261-jspb") != m.selectionHeader() {
			t.Errorf("selection header = %q", req.Header.Get("x-goog-ext-525001261-jspb"))
		}
		if req.Header.Get("x-goog-ext-525005358-jspb") != `["REQ-UUID",1]` {
			t.Errorf("uuid header = %q", req.Header.Get("x-goog-ext-525005358-jspb"))
		}
	})

	t.Run("empty model id omits headers", func(t *testing.T) {
		m := &AvailableModel{Name: "gemini-3.5-flash", ModelID: ""}
		req, err := c.newRequest(context.Background(), "f.req=x", m, "REQ-UUID")
		if err != nil {
			t.Fatalf("newRequest: %v", err)
		}
		if req.Header.Get("x-goog-ext-525001261-jspb") != "" {
			t.Errorf("selection header should be absent for an empty model id")
		}
	})
}

func TestBuildBodyImages(t *testing.T) {
	c, _ := New(testConfig(), testLogger())
	imgs := []uploadedImage{
		{ref: "/contrib_service/ttl_1d/abc", filename: "image.png"},
		{ref: "/contrib_service/ttl_1d/def", filename: "photo.jpg"},
	}
	body, _, err := c.buildBody(GenParams{Prompt: "describe"}, imgs)
	if err != nil {
		t.Fatalf("buildBody: %v", err)
	}
	inner := decodeInner(t, body)
	first, ok := inner[0].([]any)
	if !ok || len(first) < 4 {
		t.Fatalf("inner[0] = %v", inner[0])
	}
	// inner[0][3] = [[[ref], filename], ...]
	attachments, ok := first[3].([]any)
	if !ok || len(attachments) != 2 {
		t.Fatalf("inner[0][3] = %v, want 2 attachments", first[3])
	}
	a0, ok := attachments[0].([]any)
	if !ok || len(a0) != 2 {
		t.Fatalf("attachment[0] = %v, want [[ref], filename]", attachments[0])
	}
	refWrap, ok := a0[0].([]any)
	if !ok || len(refWrap) != 1 || refWrap[0] != "/contrib_service/ttl_1d/abc" {
		t.Errorf("attachment ref = %v", a0[0])
	}
	if a0[1] != "image.png" {
		t.Errorf("attachment filename = %v, want image.png", a0[1])
	}
}

func TestBuildBodyNoImages(t *testing.T) {
	c, _ := New(testConfig(), testLogger())
	body, _, _ := c.buildBody(GenParams{Prompt: "hi"}, nil)
	inner := decodeInner(t, body)
	first := inner[0].([]any)
	// inner[0][3] must be null when there are no images.
	if first[3] != nil {
		t.Errorf("inner[0][3] = %v, want null", first[3])
	}
}

// decodeInner extracts the inner payload array from a request body.
func decodeInner(t *testing.T, body string) []any {
	t.Helper()
	vals, err := url.ParseQuery(body)
	if err != nil {
		t.Fatalf("ParseQuery: %v", err)
	}
	var outer []any
	if err := json.Unmarshal([]byte(vals.Get("f.req")), &outer); err != nil {
		t.Fatalf("unmarshal f.req: %v", err)
	}
	var inner []any
	if err := json.Unmarshal([]byte(outer[1].(string)), &inner); err != nil {
		t.Fatalf("unmarshal inner: %v", err)
	}
	return inner
}

func TestBuildBodyNoAtToken(t *testing.T) {
	c, _ := New(testConfig(), testLogger())
	body, _, _ := c.buildBody(GenParams{Prompt: "hi"}, nil)
	vals, _ := url.ParseQuery(body)
	if vals.Has("at") {
		t.Errorf("at should be absent when no at token is known")
	}
}

// makeWrbLine builds a synthetic batchexecute response line carrying the given text.
func makeWrbLine(t *testing.T, text string) string {
	t.Helper()
	inner2 := []any{nil, nil, nil, nil, []any{[]any{nil, []any{text}}}}
	innerBytes, err := json.Marshal(inner2)
	if err != nil {
		t.Fatalf("marshal inner2: %v", err)
	}
	lineBytes, err := json.Marshal([]any{[]any{"wrb.fr", nil, string(innerBytes)}})
	if err != nil {
		t.Fatalf("marshal line: %v", err)
	}
	return string(lineBytes)
}

func TestParseWrbLine(t *testing.T) {
	text := strings.Repeat("hello world ", 30) // long enough to pass length gates
	line := makeWrbLine(t, text)
	if len(line) < 200 {
		t.Fatalf("synthetic line too short: %d", len(line))
	}
	got := parseWrbLine(line)
	if len(got) != 1 || got[0] != text {
		t.Errorf("parseWrbLine = %v, want [%q]", got, text)
	}
}

func TestParseWrbLineIgnoresNonWrb(t *testing.T) {
	if got := parseWrbLine(`["di",1]`); got != nil {
		t.Errorf("non-wrb line should yield nil, got %v", got)
	}
	if got := parseWrbLine(`["wrb.fr"]`); got != nil { // too short / malformed
		t.Errorf("short wrb line should yield nil, got %v", got)
	}
}

func TestExtractResponseText(t *testing.T) {
	text := strings.Repeat("The answer is 42. ", 20)
	raw := "12345\n" + makeWrbLine(t, text) + "\n67890\n"
	got := extractResponseText(raw)
	if got != strings.TrimSpace(text) {
		t.Errorf("extractResponseText = %q, want %q", got, strings.TrimSpace(text))
	}
}

func TestCleanGeminiText(t *testing.T) {
	in := "Result A\n```python?code_reference&code_event_index=0\nprint('x')\n```\nResult B"
	got := cleanGeminiText(in)
	if strings.Contains(got, "code_reference") || strings.Contains(got, "print(") {
		t.Errorf("artifact not stripped: %q", got)
	}
	if !strings.Contains(got, "Result A") || !strings.Contains(got, "Result B") {
		t.Errorf("real text removed: %q", got)
	}
}

func TestCleanGeminiTextCardContent(t *testing.T) {
	in := "Answer here.\nhttp://googleusercontent.com/card_content/12\nMore text."
	got := cleanGeminiText(in)
	if strings.Contains(got, "googleusercontent.com/card_content") {
		t.Errorf("card_content not stripped: %q", got)
	}
	if !strings.Contains(got, "Answer here.") || !strings.Contains(got, "More text.") {
		t.Errorf("real text removed: %q", got)
	}
}

func TestExtractResponseTextLongest(t *testing.T) {
	// The complete answer is the longest cumulative snapshot, even if a shorter
	// snapshot appears on a later line.
	long := makeWrbLine(t, strings.Repeat("b", 300))
	short := makeWrbLine(t, strings.Repeat("a", 250))
	got := extractResponseText(long + "\n" + short)
	if len(got) < 300 || strings.ContainsRune(got, 'a') {
		t.Errorf("expected the longest (b…) snapshot, got %d chars starting %.10q", len(got), got)
	}
}

func TestBuildLabelRegex(t *testing.T) {
	html := []byte(`window.WIZ_global_data = {"foo":"bar","cfb2h":"boq_assistant-bard-web-server_20260529.02_p0","baz":1};`)
	m := buildLabelRe.FindSubmatch(html)
	if m == nil || string(m[1]) != "boq_assistant-bard-web-server_20260529.02_p0" {
		t.Errorf("extracted %q", m)
	}
	if buildLabelRe.FindSubmatch([]byte("no build label here")) != nil {
		t.Errorf("unexpected match on label-free html")
	}
}

func TestSNlM0eRegex(t *testing.T) {
	signedIn := []byte(`...,"SNlM0e":"AOOh0PFU6lkjmU18:1780298447979","cfb2h":"boq_x"...`)
	m := snlm0eRe.FindSubmatch(signedIn)
	if m == nil {
		t.Fatalf("should match a signed-in page (SNlM0e present)")
	}
	if got := string(m[1]); got != "AOOh0PFU6lkjmU18:1780298447979" {
		t.Errorf("captured token = %q, want the SNlM0e value", got)
	}
	if snlm0eRe.Match([]byte(`anonymous page without the token`)) {
		t.Errorf("should not match an anonymous page")
	}
}

func TestCurrentAt(t *testing.T) {
	c, _ := New(testConfig(), testLogger())

	// No explicit token and nothing scraped yet → empty.
	if at := c.currentAt(); at != "" {
		t.Errorf("currentAt = %q, want empty before any token is known", at)
	}

	// A scraped SNlM0e token is used.
	c.setAt("scraped-tok")
	if at := c.currentAt(); at != "scraped-tok" {
		t.Errorf("currentAt = %q, want scraped-tok", at)
	}
}

func TestImagesSupported(t *testing.T) {
	c, _ := New(testConfig(), testLogger())
	if c.ImagesSupported() {
		t.Errorf("images must be unsupported with no at token (anonymous)")
	}
	c.setAt("snlm0e")
	if !c.ImagesSupported() {
		t.Errorf("images must be supported once an at token is known")
	}
}

func TestBuildBodyAtFromScrapedToken(t *testing.T) {
	c, _ := New(testConfig(), testLogger())
	c.setAt("snlm0e-value")
	body, _, err := c.buildBody(GenParams{Prompt: "hi"}, nil)
	if err != nil {
		t.Fatalf("buildBody: %v", err)
	}
	vals, _ := url.ParseQuery(body)
	if vals.Get("at") != "snlm0e-value" {
		t.Errorf("at = %q, want the scraped SNlM0e token", vals.Get("at"))
	}
}

func TestMakeSapisidHash(t *testing.T) {
	got := makeSapisidHash("secret")
	if !strings.HasPrefix(got, "SAPISIDHASH ") {
		t.Errorf("hash = %q, want SAPISIDHASH prefix", got)
	}
	re := regexp.MustCompile(`^SAPISIDHASH \d+_[0-9a-f]{40}$`)
	if !re.MatchString(got) {
		t.Errorf("hash format invalid: %q", got)
	}
}

func TestRuneDelta(t *testing.T) {
	cases := []struct {
		name, prev, cur, want string
	}{
		{"plain append", "Hello", "Hello, world", ", world"},
		{"empty prev", "", "abc", "abc"},
		{"no growth (equal)", "same", "same", ""},
		{"multibyte append", "Привет", "Привет, мир", ", мир"},
		{"revision not a byte-prefix", "abXdef", "abYZdef", "YZdef"},
		{"multibyte revision keeps runes intact", "café au", "café latte", "latte"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := runeDelta(tc.prev, tc.cur)
			if got != tc.want {
				t.Errorf("runeDelta(%q, %q) = %q, want %q", tc.prev, tc.cur, got, tc.want)
			}
			if !utf8.ValidString(got) {
				t.Errorf("runeDelta(%q, %q) returned invalid UTF-8: %q", tc.prev, tc.cur, got)
			}
		})
	}
}

func TestNewClampsNumericDefaults(t *testing.T) {
	// Zero and negative values for the timing/retry fields must clamp to defaults:
	// a negative RequestTimeout makes http.Client time out immediately, and a
	// negative RetryAttempts makes Generate's loop run zero times and wrap nil.
	for _, n := range []int{0, -1, -5} {
		c, err := New(Config{RequestTimeout: n, RetryAttempts: n, RetryDelaySec: n}, testLogger())
		if err != nil {
			t.Fatalf("New(n=%d): %v", n, err)
		}
		if c.cfg.RequestTimeout != DefaultRequestTimeout {
			t.Errorf("RequestTimeout=%d clamped to %d, want %d", n, c.cfg.RequestTimeout, DefaultRequestTimeout)
		}
		if c.cfg.RetryAttempts != DefaultRetryAttempts {
			t.Errorf("RetryAttempts=%d clamped to %d, want %d", n, c.cfg.RetryAttempts, DefaultRetryAttempts)
		}
		if c.cfg.RetryDelaySec != DefaultRetryDelaySec {
			t.Errorf("RetryDelaySec=%d clamped to %d, want %d", n, c.cfg.RetryDelaySec, DefaultRetryDelaySec)
		}
	}
	// Positive values are preserved.
	c, _ := New(Config{RequestTimeout: 7, RetryAttempts: 5, RetryDelaySec: 3}, testLogger())
	if c.cfg.RequestTimeout != 7 || c.cfg.RetryAttempts != 5 || c.cfg.RetryDelaySec != 3 {
		t.Errorf("positive values overwritten: timeout=%d attempts=%d delay=%d",
			c.cfg.RequestTimeout, c.cfg.RetryAttempts, c.cfg.RetryDelaySec)
	}
}
