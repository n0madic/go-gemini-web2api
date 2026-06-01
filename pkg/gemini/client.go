// Package gemini is a reusable client for Google's Gemini web backend
// (the BardChatUi batchexecute endpoints). It handles cookie/SAPISIDHASH auth,
// dynamic build-label and model-catalog resolution, cookie rotation, image
// uploads, and blocking or streaming text generation.
//
// A typical consumer builds a Config, calls New, optionally resolves the build
// label / auth / model list at startup, and then calls Generate or
// GenerateStream:
//
//	c, err := gemini.New(gemini.Config{CookieFile: "cookie.txt"}, logger)
//	c.CheckAuth(ctx)
//	c.ResolveModels(ctx)
//	m, _ := c.ResolveModelOrDefault("gemini-3.5-flash")
//	text, err := c.Generate(ctx, m.Params("Hello"))
package gemini

import (
	"bufio"
	"context"
	"crypto/sha1"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/n0madic/go-gemini-web2api/pkg/util"
)

// geminiBaseURL is the public Gemini host serving the BardChatUi batchexecute endpoint.
const geminiBaseURL = "https://gemini.google.com"

// userAgent mirrors a desktop Chrome UA, as the reference does.
const userAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

// codeArtifactRe strips internal code-execution artifacts that Gemini embeds in
// streamed text (e.g. ```python?code_reference&code_event_index=0 ... ```).
var codeArtifactRe = regexp.MustCompile("(?s)```(?:python|javascript|text)\\?code_(?:reference|stdout)&code_event_index=\\d+\\n.*?```\\n?")

// cardContentRe strips googleusercontent card-content reference URLs.
var cardContentRe = regexp.MustCompile(`http://googleusercontent\.com/card_content/\d+\n?`)

// ErrEmptyResponse indicates Gemini returned a 200 with no extractable text,
// typically because the prompt exceeded the web endpoint's size limit. Generate
// returns it (unwrapped) so callers can detect this case with errors.Is.
var ErrEmptyResponse = errors.New("empty response from Gemini (prompt may exceed the web endpoint size limit)")

// buildLabelRe extracts the build label (bl) from WIZ_global_data on the Gemini page.
var buildLabelRe = regexp.MustCompile(`"cfb2h":"(boq_[^"]+)"`)

// snlm0eRe captures the SNlM0e (XSRF/"at") token, which the Gemini page embeds
// only for a signed-in session: its presence proves the cookie authenticates, and
// its value is required as the "at" form field on authenticated StreamGenerate calls.
var snlm0eRe = regexp.MustCompile(`"SNlM0e":"([^"]+)"`)

// buildLabelTTL is how long an auto-resolved build label is reused before refresh.
const buildLabelTTL = 30 * time.Minute

// Client talks to the Gemini web backend. It is safe for concurrent use.
type Client struct {
	cfg  Config
	http *http.Client
	log  *slog.Logger

	// cookie file cache, keyed by file mtime to avoid re-reading every request.
	cookieMu     sync.Mutex
	cookieStr    string
	cookieSAPI   string
	cookieMtime  time.Time
	cookieLoaded bool

	// Build label, optionally auto-resolved from the Gemini page.
	blMu         sync.Mutex
	bl           string
	blAuto       bool
	blAt         time.Time
	blRefreshing bool

	// XSRF "at" token (SNlM0e), scraped from the signed-in page. Required on
	// authenticated StreamGenerate requests; anonymous requests omit it.
	atMu    sync.Mutex
	atToken string

	// Model registry, populated from the listModels RPC (or the static fallback
	// set). modelOrder preserves a stable listing order.
	modelMu          sync.Mutex
	models           map[string]*AvailableModel
	modelOrder       []string
	modelsSource     string // "dynamic" | "fallback" | "static"
	modelsAt         time.Time
	modelsRefreshing bool
}

// New builds a Client with an HTTP client honouring the configured timeout and
// proxy. Any Config field left at its zero value is filled with the matching
// Default* constant (see Config.applyDefaults), so gemini.Config{} — optionally
// with just a cookie — yields a working client. A nil logger defaults to
// slog.Default().
func New(cfg Config, logger *slog.Logger) (*Client, error) {
	cfg.applyDefaults()
	if logger == nil {
		logger = slog.Default()
	}
	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
	}
	if cfg.Proxy != "" {
		pu, err := url.Parse(cfg.Proxy)
		if err != nil {
			return nil, fmt.Errorf("invalid proxy URL: %w", err)
		}
		transport.Proxy = http.ProxyURL(pu)
	}
	return &Client{
		cfg: cfg,
		http: &http.Client{
			Timeout:   time.Duration(cfg.RequestTimeout) * time.Second,
			Transport: transport,
		},
		log:    logger,
		bl:     cfg.GeminiBL,
		blAuto: cfg.GeminiBLAuto,
	}, nil
}

// GenParams carries everything needed for a single generation request.
type GenParams struct {
	Prompt string
	Images []InputImage    // images to attach (uploaded just before generation)
	Model  *AvailableModel // selected model (drives the selection headers)
}

// InputImage is an image to attach to a generation request. Exactly one of Data
// (inline bytes already decoded from base64) or URL (a remote http(s) image to
// fetch at upload time) is populated.
type InputImage struct {
	Data     []byte
	URL      string
	Filename string
}

// buildBody constructs the urlencoded request body for StreamGenerate. The Gemini
// backend expects f.req to be [null, "<inner-json-string>"] where the inner payload
// is a sparse 69-element array with specific indices populated. Attached images
// (already uploaded) are referenced at inner[0][3]. The model is selected via HTTP
// headers (see AvailableModel.headers), not the payload; inner[17] is a constant.
//
// It also returns the per-request UUID (inner[59]); the caller must echo it in the
// x-goog-ext-525005358-jspb selection header so the two stay in sync.
func (c *Client) buildBody(p GenParams, images []uploadedImage) (body, reqUUID string, err error) {
	reqUUID = strings.ToUpper(uuid4())
	inner := make([]any, 69)
	inner[0] = []any{p.Prompt, 0, nil, imageRefsField(images), nil, nil, 0}
	inner[1] = []any{"en"}
	inner[2] = []any{"", "", "", nil, nil, nil, nil, nil, nil, ""}
	inner[6] = []any{1}
	inner[7] = 1
	inner[10] = 1
	inner[11] = 0
	inner[17] = []any{[]any{0}} // constant (reasoning level is not exposed)
	inner[18] = 0
	inner[27] = 1
	inner[30] = []any{4}
	inner[41] = []any{1}
	inner[53] = 0
	inner[59] = reqUUID
	inner[61] = []any{} // must serialize as [] not null
	inner[68] = 2

	innerJSON, err := util.MarshalNoEscape(inner)
	if err != nil {
		return "", "", err
	}
	outerJSON, err := util.MarshalNoEscape([]any{nil, string(innerJSON)})
	if err != nil {
		return "", "", err
	}

	vals := url.Values{}
	vals.Set("f.req", string(outerJSON))
	if at := c.currentAt(); at != "" {
		vals.Set("at", at)
	}
	return vals.Encode(), reqUUID, nil
}

// streamURL builds the StreamGenerate URL with a per-request id.
func (c *Client) streamURL() string {
	reqid := time.Now().Unix() % 1000000
	prefix := c.accountPrefix()
	return fmt.Sprintf(
		"%s%s/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"+
			"?bl=%s&hl=en&_reqid=%d&rt=c",
		geminiBaseURL, prefix, url.QueryEscape(c.CurrentBL()), reqid)
}

// CurrentBL returns the active build label, kicking off a background refresh when
// auto-resolution is enabled and the cached value has gone stale.
func (c *Client) CurrentBL() string {
	c.blMu.Lock()
	bl := c.bl
	stale := c.blAuto && !c.blRefreshing && (c.blAt.IsZero() || time.Since(c.blAt) > buildLabelTTL)
	if stale {
		c.blRefreshing = true
	}
	c.blMu.Unlock()
	if stale {
		go c.fetchAndStoreBL(context.Background())
	}
	return bl
}

// ResolveBuildLabel synchronously resolves the build label (used at startup). It
// is a no-op when auto-resolution is disabled.
func (c *Client) ResolveBuildLabel(ctx context.Context) {
	if !c.blAuto {
		return
	}
	c.blMu.Lock()
	if c.blRefreshing {
		c.blMu.Unlock()
		return
	}
	c.blRefreshing = true
	c.blMu.Unlock()
	c.fetchAndStoreBL(ctx)
}

// fetchAndStoreBL fetches the build label and stores it, recording the attempt
// time even on failure so refreshes are not retried too aggressively.
func (c *Client) fetchAndStoreBL(ctx context.Context) {
	bl := c.fetchBuildLabel(ctx)
	c.blMu.Lock()
	if bl != "" {
		c.bl = bl
	}
	c.blAt = time.Now()
	c.blRefreshing = false
	c.blMu.Unlock()
	if bl != "" {
		c.log.Info("resolved gemini build label", "bl", bl)
	} else {
		c.log.Warn("could not resolve gemini build label; using fallback", "fallback", c.cfg.GeminiBL)
	}
}

// fetchAppPage GETs the Gemini /app page (with the cookie when present); it is the
// shared source for scraping the build label and the SNlM0e XSRF token. The body is
// capped at 4MB.
func (c *Client) fetchAppPage(ctx context.Context) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, geminiBaseURL+c.accountPrefix()+"/app", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	if cookie, _ := c.loadCookie(); cookie != "" {
		req.Header.Set("Cookie", cookie)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(io.LimitReader(resp.Body, 4<<20))
}

// fetchBuildLabel scrapes the current build label from the Gemini app page.
func (c *Client) fetchBuildLabel(ctx context.Context) string {
	html, err := c.fetchAppPage(ctx)
	if err != nil {
		c.log.Warn("build label fetch failed", "err", err)
		return ""
	}
	if m := buildLabelRe.FindSubmatch(html); m != nil {
		return string(m[1])
	}
	return ""
}

// accountPrefix returns the /u/N path segment for non-default Google accounts.
func (c *Client) accountPrefix() string {
	if c.cfg.AuthUser == "" {
		return ""
	}
	return "/u/" + c.cfg.AuthUser
}

// CheckAuth reports whether the configured cookie establishes a signed-in Gemini
// session, detected via the SNlM0e token (issued only to authenticated sessions).
// It returns a short human-readable status and logs a warning when a cookie is
// present but does not authenticate. The token value itself is never logged.
func (c *Client) CheckAuth(ctx context.Context) string {
	cookie, _ := c.loadCookie()
	if cookie == "" {
		return "anonymous"
	}
	html, err := c.fetchAppPage(ctx)
	if err != nil {
		c.log.Warn("cookie auth check failed", "err", err)
		return "unknown (fetch failed)"
	}
	if m := snlm0eRe.FindSubmatch(html); m != nil {
		c.setAt(string(m[1])) // authenticated requests must echo this XSRF token back
		c.log.Info("gemini cookie authenticated")
		return "authenticated"
	}
	c.log.Warn("gemini cookie is NOT authenticating — requests run anonymously",
		"hint", "the session needs __Secure-1PSIDTS (it expires; re-export the cookie)")
	return "NOT authenticated"
}

// currentAt returns the XSRF "at" token to send with StreamGenerate: the SNlM0e
// value scraped from the signed-in page (refreshed after each cookie rotation, as
// it rotates with the session). Empty for anonymous sessions, which need no token.
func (c *Client) currentAt() string {
	c.atMu.Lock()
	defer c.atMu.Unlock()
	return c.atToken
}

// setAt caches the scraped SNlM0e XSRF token.
func (c *Client) setAt(token string) {
	c.atMu.Lock()
	c.atToken = token
	c.atMu.Unlock()
}

// ImagesSupported reports whether image attachments can be used. They require an
// authenticated session (a non-empty XSRF "at" token); anonymous sessions — and
// sessions whose cookie no longer authenticates — cannot attach images.
func (c *Client) ImagesSupported() bool {
	return c.currentAt() != ""
}

// refreshAtToken re-scrapes the SNlM0e XSRF token from the signed-in page and
// caches it. Called after a cookie rotation so the token stays in sync with the
// refreshed session.
func (c *Client) refreshAtToken(ctx context.Context) {
	if cookie, _ := c.loadCookie(); cookie == "" {
		return
	}
	html, err := c.fetchAppPage(ctx)
	if err != nil {
		return
	}
	if m := snlm0eRe.FindSubmatch(html); m != nil {
		c.setAt(string(m[1]))
	}
}

// newRequest assembles a POST request with all required headers and auth. When a
// model with a known id is supplied, the four model-selection headers are set
// (reqUUID must match inner[59] of the body). A nil model or an empty model id
// (the fallback/anonymous case) omits them so the request uses the account default.
func (c *Client) newRequest(ctx context.Context, body string, model *AvailableModel, reqUUID string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.streamURL(), strings.NewReader(body))
	if err != nil {
		return nil, err
	}
	prefix := c.accountPrefix()
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Origin", geminiBaseURL)
	req.Header.Set("Referer", geminiBaseURL+prefix+"/app")
	req.Header.Set("X-Same-Domain", "1")
	req.Header.Set("User-Agent", userAgent)
	if prefix != "" {
		req.Header.Set("X-Goog-AuthUser", c.cfg.AuthUser)
	}
	if model != nil && model.ModelID != "" {
		for k, v := range model.headers(reqUUID) {
			req.Header.Set(k, v)
		}
	}

	cookie, sapisid := c.loadCookie()
	if cookie != "" {
		req.Header.Set("Cookie", cookie)
	}
	if sapisid != "" {
		req.Header.Set("Authorization", makeSapisidHash(sapisid))
	}
	return req, nil
}

// Generate sends a prompt and returns the final response text (blocking, with retry).
func (c *Client) Generate(ctx context.Context, p GenParams) (string, error) {
	images := c.uploadImages(ctx, p.Images)
	body, reqUUID, err := c.buildBody(p, images)
	if err != nil {
		return "", err
	}

	var lastErr error
	for attempt := 0; attempt < c.cfg.RetryAttempts; attempt++ {
		req, err := c.newRequest(ctx, body, p.Model, reqUUID)
		if err != nil {
			return "", err
		}
		resp, err := c.http.Do(req)
		if err == nil {
			raw, readErr := io.ReadAll(resp.Body)
			resp.Body.Close()
			switch {
			case readErr != nil:
				err = readErr
			default:
				// An empty body is treated as a (retryable) failure rather than a
				// silent empty answer.
				if text := extractResponseText(string(raw)); text != "" {
					return text, nil
				}
				err = ErrEmptyResponse
			}
		}
		lastErr = err
		if attempt < c.cfg.RetryAttempts-1 {
			c.log.Warn("retrying gemini request",
				"attempt", attempt+1, "max", c.cfg.RetryAttempts, "err", err)
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(time.Duration(c.cfg.RetryDelaySec) * time.Second):
			}
		}
	}
	// ErrEmptyResponse is already self-descriptive; don't double-wrap it.
	if errors.Is(lastErr, ErrEmptyResponse) {
		return "", ErrEmptyResponse
	}
	return "", fmt.Errorf("gemini request failed: %w", lastErr)
}

// GenerateStream sends a prompt and invokes emit for each incremental text delta.
// Gemini sends cumulative text per chunk, so deltas are computed against prevText.
func (c *Client) GenerateStream(ctx context.Context, p GenParams, emit func(delta string)) error {
	images := c.uploadImages(ctx, p.Images)
	body, reqUUID, err := c.buildBody(p, images)
	if err != nil {
		return err
	}
	req, err := c.newRequest(ctx, body, p.Model, reqUUID)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("gemini request failed: %w", err)
	}
	defer resp.Body.Close()

	// ReadString grows its buffer as needed, unlike bufio.Scanner which caps at 64KB;
	// batchexecute lines routinely exceed that.
	reader := bufio.NewReader(resp.Body)
	prevText := ""
	for {
		line, readErr := reader.ReadString('\n')
		if line != "" {
			if text, ok := latestText(line); ok && len(text) > len(prevText) {
				delta := cleanGeminiText(runeDelta(prevText, text))
				prevText = text
				if delta != "" {
					emit(delta)
				}
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				return nil
			}
			return readErr
		}
	}
}

// runeDelta returns the part of cur that follows the longest rune-aligned prefix
// it shares with prev. Gemini streams cumulative snapshots, so cur is normally a
// byte-prefix extension of prev and the delta is simply its tail; but if a
// snapshot revises earlier text (so prev is not a byte-prefix of cur), slicing at
// len(prev) could land in the middle of a multi-byte rune. Diffing on rune
// boundaries keeps every emitted delta valid UTF-8.
func runeDelta(prev, cur string) string {
	if strings.HasPrefix(cur, prev) {
		// Fast path: prev ends on a rune boundary within cur, so the tail is safe.
		return cur[len(prev):]
	}
	i := 0
	for i < len(prev) && i < len(cur) {
		rp, size := utf8.DecodeRuneInString(prev[i:])
		rc, _ := utf8.DecodeRuneInString(cur[i:])
		if rp != rc {
			break
		}
		i += size
	}
	return cur[i:]
}

// extractResponseText parses a full StreamGenerate response and returns the final
// text. Gemini streams cumulative snapshots, so the longest candidate is the
// complete answer (more robust than picking the last non-empty one).
func extractResponseText(raw string) string {
	longest := ""
	for _, line := range strings.Split(raw, "\n") {
		for _, t := range parseWrbLine(line) {
			if len(t) > len(longest) {
				longest = t
			}
		}
	}
	return cleanGeminiText(longest)
}

// latestText returns the longest text candidate from a single streamed line.
func latestText(line string) (string, bool) {
	texts := parseWrbLine(line)
	best := ""
	for _, t := range texts {
		if len(t) > len(best) {
			best = t
		}
	}
	return best, best != ""
}

// parseWrbLine extracts text strings from a batchexecute "wrb.fr" response line.
// The line is JSON like [["wrb.fr", _, "<inner-json>", ...], ...]; the inner JSON
// nests the model text at inner2[4][*][1][*].
func parseWrbLine(line string) []string {
	if !strings.Contains(line, `"wrb.fr"`) || len(line) < 200 {
		return nil
	}
	var arr []any
	if err := json.Unmarshal([]byte(line), &arr); err != nil {
		return nil
	}
	first, ok := indexSlice(arr, 0)
	if !ok {
		return nil
	}
	innerStr, ok := indexString(first, 2)
	if !ok || len(innerStr) < 50 {
		return nil
	}
	var inner2 any
	if err := json.Unmarshal([]byte(innerStr), &inner2); err != nil {
		return nil
	}
	inner2Slice, ok := inner2.([]any)
	if !ok || len(inner2Slice) <= 4 {
		return nil
	}
	parts, ok := inner2Slice[4].([]any)
	if !ok {
		return nil
	}
	var out []string
	for _, part := range parts {
		ps, ok := part.([]any)
		if !ok || len(ps) <= 1 {
			continue
		}
		texts, ok := ps[1].([]any)
		if !ok {
			continue
		}
		for _, t := range texts {
			if s, ok := t.(string); ok && s != "" {
				out = append(out, s)
			}
		}
	}
	return out
}

// indexSlice returns arr[i] as a []any if possible.
func indexSlice(arr []any, i int) ([]any, bool) {
	if i < 0 || i >= len(arr) {
		return nil, false
	}
	s, ok := arr[i].([]any)
	return s, ok
}

// indexString returns arr[i] as a string if possible.
func indexString(arr []any, i int) (string, bool) {
	if i < 0 || i >= len(arr) {
		return "", false
	}
	s, ok := arr[i].(string)
	return s, ok
}

// cleanGeminiText removes code-execution and card-content artifacts and trims
// surrounding whitespace.
func cleanGeminiText(text string) string {
	text = codeArtifactRe.ReplaceAllString(text, "")
	text = cardContentRe.ReplaceAllString(text, "")
	return strings.TrimSpace(text)
}

// ─── Cookie / SAPISIDHASH auth ───────────────────────────────────────────────

// loadCookie resolves the cookie string and its SAPISID (inline Cookie takes
// precedence over the cookie file). The SAPISID is parsed from the cookie string
// itself. An inline cookie is rotated in memory, so a rotated value (cached in
// cookieStr) is preferred over the original config value once available.
func (c *Client) loadCookie() (cookie, sapisid string) {
	if c.cfg.Cookie != "" {
		c.cookieMu.Lock()
		cached, cachedSapi := c.cookieStr, c.cookieSAPI
		c.cookieMu.Unlock()
		if cached != "" {
			return cached, cachedSapi
		}
		return c.cfg.Cookie, cookieValue(c.cfg.Cookie, cookieSAPISID)
	}
	if c.cfg.CookieFile == "" {
		return "", ""
	}
	return c.loadCookieFile()
}

// loadCookieFile reads the cookie file, caching the parsed result by mtime so
// the file is not re-read and re-parsed on every request.
func (c *Client) loadCookieFile() (cookie, sapisid string) {
	c.cookieMu.Lock()
	defer c.cookieMu.Unlock()

	info, err := os.Stat(c.cfg.CookieFile)
	if err != nil {
		c.log.Error("cookie stat failed", "path", c.cfg.CookieFile, "err", err)
		return c.cookieStr, c.cookieSAPI // fall back to last good value
	}
	if c.cookieLoaded && info.ModTime().Equal(c.cookieMtime) {
		return c.cookieStr, c.cookieSAPI
	}

	data, err := os.ReadFile(c.cfg.CookieFile)
	if err != nil {
		c.log.Error("cookie load failed", "path", c.cfg.CookieFile, "err", err)
		return c.cookieStr, c.cookieSAPI
	}
	cookie = strings.TrimSpace(string(data))
	sapisid = cookieValue(cookie, cookieSAPISID)
	c.cookieStr, c.cookieSAPI = cookie, sapisid
	c.cookieMtime, c.cookieLoaded = info.ModTime(), true
	return cookie, sapisid
}

// makeSapisidHash builds the SAPISIDHASH Authorization header value.
func makeSapisidHash(sapisid string) string {
	ts := time.Now().Unix()
	sum := sha1.Sum(fmt.Appendf(nil, "%d %s %s", ts, sapisid, geminiBaseURL))
	return fmt.Sprintf("SAPISIDHASH %d_%x", ts, sum)
}
