package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"
)

// listModelsRPC is the batchexecute RPC id that returns the account-specific
// model catalog (and tier/capability flags).
const listModelsRPC = "otAQ7b"

// modelsTTL is how long a fetched dynamic model list is reused before a
// background refresh is triggered.
const modelsTTL = 30 * time.Minute

// genericModelHeader is the x-goog-ext-525001261-jspb value used on the model
// listing RPC (no specific model id, just the [4] suffix the web client sends).
const genericModelHeader = "[1,null,null,null,null,null,null,null,[4]]"

// AvailableModel is a model exposed to the account, as returned by the listModels
// RPC (or the static fallback). Capacity/CapacityField encode the account tier and
// are sent in the model-selection header on each generation request.
type AvailableModel struct {
	Name          string // public slug, e.g. gemini-3.5-flash
	ModelID       string // hex id used in the selection header (empty for fallback)
	DisplayName   string // short label from the RPC, e.g. "Fast"
	Description   string
	Capacity      int
	CapacityField int
}

// Params builds the base GenParams selecting this model. Callers then attach
// images before calling Generate/GenerateStream.
func (m *AvailableModel) Params(prompt string) GenParams {
	return GenParams{Prompt: prompt, Model: m}
}

// selectionHeader builds the x-goog-ext-525001261-jspb value that selects this
// model. For capacity_field 13 the capacity is preceded by an extra null slot
// (mirrors the reference web client).
func (m *AvailableModel) selectionHeader() string {
	tail := strconv.Itoa(m.Capacity)
	if m.CapacityField == 13 {
		tail = "null," + tail
	}
	return fmt.Sprintf("[1,null,null,null,%q,null,null,0,[4],null,null,%s]", m.ModelID, tail)
}

// headers returns the four HTTP headers that select this model on a
// StreamGenerate request. reqUUID must equal inner[59] of the request body.
func (m *AvailableModel) headers(reqUUID string) map[string]string {
	return map[string]string{
		"x-goog-ext-525001261-jspb": m.selectionHeader(),
		"x-goog-ext-73010989-jspb":  "[0]",
		"x-goog-ext-73010990-jspb":  "[0]",
		"x-goog-ext-525005358-jspb": fmt.Sprintf("[%q,1]", reqUUID),
	}
}

// ─── Registry accessors (on Client) ──────────────────────────────────────────

// ResolveModel looks up a model by name, substituting the configured default for
// an empty name.
func (c *Client) ResolveModel(name string) (*AvailableModel, error) {
	if name == "" {
		if m := c.defaultModel(); m != nil {
			return m, nil
		}
		return nil, errors.New("no models available")
	}
	c.modelMu.Lock()
	m, ok := c.models[name]
	c.modelMu.Unlock()
	if !ok {
		return nil, fmt.Errorf("unknown model: %q", name)
	}
	return m, nil
}

// ResolveModelOrDefault resolves a model name, falling back to the configured
// default (then the first registered model) when the name is unknown — used for
// foreign ids like "claude-*" from Claude Code.
func (c *Client) ResolveModelOrDefault(name string) *AvailableModel {
	if m, err := c.ResolveModel(name); err == nil {
		return m
	}
	return c.defaultModel()
}

// defaultModel returns the configured default model, or the first registered
// model when the default is absent (e.g. an account whose dynamic list does not
// include the configured default name).
func (c *Client) defaultModel() *AvailableModel {
	c.modelMu.Lock()
	defer c.modelMu.Unlock()
	if m, ok := c.models[c.cfg.DefaultModel]; ok {
		return m
	}
	if len(c.modelOrder) > 0 {
		return c.models[c.modelOrder[0]]
	}
	return nil
}

// ListModels returns the registered models in listing order, triggering a
// background refresh when the cached list has gone stale.
func (c *Client) ListModels() []*AvailableModel {
	c.maybeRefreshModels()
	c.modelMu.Lock()
	defer c.modelMu.Unlock()
	out := make([]*AvailableModel, 0, len(c.modelOrder))
	for _, name := range c.modelOrder {
		if m, ok := c.models[name]; ok {
			out = append(out, m)
		}
	}
	return out
}

// ModelNames returns the registered model names in listing order.
func (c *Client) ModelNames() []string {
	c.maybeRefreshModels()
	c.modelMu.Lock()
	defer c.modelMu.Unlock()
	return append([]string(nil), c.modelOrder...)
}

// ModelsSourceLabel reports where the current list came from: "dynamic" (RPC),
// "fallback" (static), "static" (set via SetModels), or "" before the registry is
// populated.
func (c *Client) ModelsSourceLabel() string {
	c.modelMu.Lock()
	defer c.modelMu.Unlock()
	return c.modelsSource
}

// SetModels installs an explicit model catalog, bypassing the network. The source
// label is reported as "static". This lets a consumer seed the registry offline
// (e.g. in tests) using the public API.
func (c *Client) SetModels(models []*AvailableModel) {
	c.storeModels(models, "static")
}

// maybeRefreshModels kicks off a background dynamic refresh when the cached list
// is older than modelsTTL (mirrors CurrentBL).
func (c *Client) maybeRefreshModels() {
	c.modelMu.Lock()
	stale := !c.modelsRefreshing && (c.modelsAt.IsZero() || time.Since(c.modelsAt) > modelsTTL)
	if stale {
		c.modelsRefreshing = true
	}
	c.modelMu.Unlock()
	if stale {
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			c.fetchModels(ctx)
		}()
	}
}

// storeModels atomically replaces the registry with the given models, preserving
// their order and recording the source and fetch time.
func (c *Client) storeModels(models []*AvailableModel, source string) {
	reg := make(map[string]*AvailableModel, len(models))
	order := make([]string, 0, len(models))
	for _, m := range models {
		if _, dup := reg[m.Name]; dup {
			continue
		}
		reg[m.Name] = m
		order = append(order, m.Name)
	}
	c.modelMu.Lock()
	c.models = reg
	c.modelOrder = order
	c.modelsSource = source
	c.modelsAt = time.Now()
	c.modelsRefreshing = false
	c.modelMu.Unlock()
}

// ─── Fetching ────────────────────────────────────────────────────────────────

// ResolveModels populates the registry at startup (synchronous) and logs the
// result once. It installs the dynamic list when available, otherwise the static
// fallback set. When the configured default model is not in the resolved list
// (e.g. the account renamed or dropped it), it warns and names the substitute, so
// a model rename is observable rather than silent.
func (c *Client) ResolveModels(ctx context.Context) {
	c.fetchModels(ctx)
	c.modelMu.Lock()
	count, source := len(c.modelOrder), c.modelsSource
	_, haveDefault := c.models[c.cfg.DefaultModel]
	var substitute string
	if !haveDefault && len(c.modelOrder) > 0 {
		substitute = c.modelOrder[0]
	}
	c.modelMu.Unlock()
	c.log.Info("resolved gemini model list", "count", count, "source", source)
	if c.cfg.DefaultModel != "" && !haveDefault && substitute != "" {
		c.log.Warn("configured default model not in account list; using first available instead",
			"configured", c.cfg.DefaultModel, "using", substitute,
			"hint", "set GEMINI_DEFAULT_MODEL to a name from /v1/models")
	}
}

// fetchModels queries the dynamic model list and replaces the registry. On
// failure it keeps an existing good list rather than downgrading; only when the
// registry is still empty (startup) does it install the fallback set. Routine
// refreshes log at Debug; the one-time startup summary is logged by ResolveModels.
func (c *Client) fetchModels(ctx context.Context) {
	models, err := c.dynamicModels(ctx)
	if err == nil {
		c.storeModels(models, "dynamic")
		c.log.Debug("refreshed gemini model list", "count", len(models), "source", "dynamic")
		return
	}

	c.modelMu.Lock()
	have := len(c.modelOrder) > 0
	c.modelsRefreshing = false
	c.modelMu.Unlock()
	if have {
		// A transient refresh failure must not downgrade a good dynamic list.
		c.log.Debug("model list refresh failed; keeping current list", "err", err)
		return
	}
	if c.currentAt() == "" {
		c.log.Warn("no authenticated session for model list; using fallback model list")
	} else {
		c.log.Warn("dynamic model list unavailable; using fallback model list", "err", err)
	}
	c.storeModels(fallbackModels(c.cfg.DefaultModel), "fallback")
}

// dynamicModels fetches and parses the account's model catalog via the
// listModels RPC. It requires an authenticated session.
func (c *Client) dynamicModels(ctx context.Context) ([]*AvailableModel, error) {
	if c.currentAt() == "" {
		return nil, errors.New("session is not authenticated")
	}
	raw, err := c.batchExecute(ctx, listModelsRPC, "[]")
	if err != nil {
		return nil, err
	}
	inner, ok := parseBatchResponse(raw, listModelsRPC)
	if !ok {
		return nil, fmt.Errorf("%s: wrb.fr entry not found in response", listModelsRPC)
	}
	return parseModels(inner)
}

// fallbackModels returns the model set used when the dynamic list cannot be
// fetched (anonymous session, an expired cookie, or an RPC failure). In that mode
// there are no real model ids to select anything other than the backend's default
// (Flash) model, so listing additional names would be fictitious — the catalog is
// honestly just the configured default, and requesting any other name returns 400.
func fallbackModels(defaultModel string) []*AvailableModel {
	if defaultModel == "" {
		defaultModel = "gemini-3.5-flash"
	}
	return []*AvailableModel{{
		Name:          defaultModel,
		Description:   "Default model (the only model available without an authenticated cookie)",
		Capacity:      1,
		CapacityField: 12,
	}}
}

// ─── batchexecute + parsing ──────────────────────────────────────────────────

// batchExecute POSTs a single batchexecute RPC and returns the raw response
// body. It reuses the same cookie / SAPISIDHASH / build-label / at-token
// machinery as StreamGenerate. The call is authenticated; an anonymous session
// (no "at" token) will not receive account data.
func (c *Client) batchExecute(ctx context.Context, rpcid, payload string) (string, error) {
	reqid := time.Now().Unix() % 1000000
	prefix := c.accountPrefix()
	q := url.Values{}
	q.Set("rpcids", rpcid)
	q.Set("source-path", "/app")
	q.Set("bl", c.CurrentBL())
	q.Set("hl", "en")
	q.Set("_reqid", strconv.FormatInt(reqid, 10))
	q.Set("rt", "c")
	u := geminiBaseURL + prefix + "/_/BardChatUi/data/batchexecute?" + q.Encode()

	freq, err := json.Marshal([]any{[]any{[]any{rpcid, payload, nil, "generic"}}})
	if err != nil {
		return "", err
	}
	form := url.Values{}
	form.Set("f.req", string(freq))
	if at := c.currentAt(); at != "" {
		form.Set("at", at)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, strings.NewReader(form.Encode()))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded;charset=utf-8")
	req.Header.Set("Origin", geminiBaseURL)
	req.Header.Set("Referer", geminiBaseURL+prefix+"/app")
	req.Header.Set("X-Same-Domain", "1")
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("x-goog-ext-525001261-jspb", genericModelHeader)
	if prefix != "" {
		req.Header.Set("X-Goog-AuthUser", c.cfg.AuthUser)
	}
	cookie, sapisid := c.loadCookie()
	if cookie != "" {
		req.Header.Set("Cookie", cookie)
	}
	if sapisid != "" {
		req.Header.Set("Authorization", makeSapisidHash(sapisid))
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("batchexecute %s: http %d", rpcid, resp.StatusCode)
	}
	return string(body), nil
}

// parseBatchResponse strips the )]}' XSSI guard, decodes the consecutive JSON
// values of a batchexecute response, and returns the inner JSON payload of the
// ["wrb.fr","<rpcid>",...] entry.
func parseBatchResponse(raw, rpcid string) (string, bool) {
	raw = strings.TrimSpace(raw)
	raw = strings.TrimPrefix(raw, ")]}'")
	dec := json.NewDecoder(strings.NewReader(raw))
	for {
		var v any
		if err := dec.Decode(&v); err != nil {
			break // EOF or an unreadable length token; stop scanning
		}
		if s, ok := searchWrb(v, rpcid); ok {
			return s, true
		}
	}
	return "", false
}

// searchWrb recursively looks for a ["wrb.fr","<rpcid>","<inner-json>",…] entry
// and returns the inner JSON string.
func searchWrb(v any, rpcid string) (string, bool) {
	arr, ok := v.([]any)
	if !ok {
		return "", false
	}
	if len(arr) >= 3 {
		if a0, _ := arr[0].(string); a0 == "wrb.fr" {
			if a1, _ := arr[1].(string); a1 == rpcid {
				if s, ok := arr[2].(string); ok {
					return s, true
				}
			}
		}
	}
	for _, child := range arr {
		if s, ok := searchWrb(child, rpcid); ok {
			return s, true
		}
	}
	return "", false
}

// parseModels parses the inner JSON of the listModels RPC into the available
// model list. body[15] is the model array; body[16]/body[17] are the tier and
// capability flags fed to computeCapacity.
func parseModels(innerJSON string) ([]*AvailableModel, error) {
	var body []any
	if err := json.Unmarshal([]byte(innerJSON), &body); err != nil {
		return nil, err
	}
	tier := floatList(body, 16)
	capFlags := floatList(body, 17)
	capacity, capField := computeCapacity(tier, capFlags)

	rawModels := anyList(body, 15)
	out := make([]*AvailableModel, 0, len(rawModels))
	seen := make(map[string]bool)
	for _, rm := range rawModels {
		ml, ok := rm.([]any)
		if !ok {
			continue
		}
		id := strAt(ml, 0)
		if id == "" {
			continue
		}
		name := modelSlug(strAt(ml, 11), strAt(ml, 1), id)
		if seen[name] {
			continue
		}
		seen[name] = true
		out = append(out, &AvailableModel{
			Name:          name,
			ModelID:       id,
			DisplayName:   strAt(ml, 1),
			Description:   strAt(ml, 2),
			Capacity:      capacity,
			CapacityField: capField,
		})
	}
	if len(out) == 0 {
		return nil, errors.New("no models in listModels response")
	}
	return out, nil
}

// computeCapacity maps the account tier/capability flags onto Gemini's
// (capacity, capacity_field) pair (ported from HanaokaYuzu's compute_capacity).
func computeCapacity(tier, capFlags []float64) (capacity, capacityField int) {
	switch {
	case slices.Contains(tier, 21):
		return 1, 13
	case slices.Contains(tier, 22):
		return 2, 13
	case slices.Contains(capFlags, 115):
		return 4, 12 // Plus
	case slices.Contains(tier, 16) || slices.Contains(capFlags, 106):
		return 3, 12 // Pro (uncommon)
	case slices.Contains(tier, 8) || (!slices.Contains(capFlags, 106) && slices.Contains(capFlags, 19)):
		return 2, 12 // Pro
	default:
		return 1, 12 // Free
	}
}

// modelSlug derives the public model name: from the versioned name ([11]) when
// present, else the short label ([1]), else gemini-<id>.
func modelSlug(versioned, label, id string) string {
	if strings.TrimSpace(versioned) != "" {
		return slugifyModelName(versioned)
	}
	if strings.TrimSpace(label) != "" {
		return slugifyModelName(label)
	}
	return "gemini-" + id
}

// slugifyModelName turns a versioned model name (e.g. "3.1 Flash-Lite") into a
// slug ("gemini-3.1-flash-lite"): lowercased, whitespace collapsed to single
// hyphens, dots and existing hyphens preserved.
func slugifyModelName(versioned string) string {
	fields := strings.Fields(strings.ToLower(versioned))
	return "gemini-" + strings.Join(fields, "-")
}

// ─── Small JSON-array accessors ──────────────────────────────────────────────

// floatList returns arr[i] as a []float64 (non-numeric elements skipped).
func floatList(arr []any, i int) []float64 {
	l, ok := indexAny(arr, i).([]any)
	if !ok {
		return nil
	}
	out := make([]float64, 0, len(l))
	for _, x := range l {
		if f, ok := x.(float64); ok {
			out = append(out, f)
		}
	}
	return out
}

// anyList returns arr[i] as a []any (nil when absent or not an array).
func anyList(arr []any, i int) []any {
	l, _ := indexAny(arr, i).([]any)
	return l
}

// strAt returns arr[i] as a string (empty when absent or not a string).
func strAt(arr []any, i int) string {
	s, _ := indexAny(arr, i).(string)
	return s
}

// indexAny returns arr[i] or nil when out of range.
func indexAny(arr []any, i int) any {
	if i < 0 || i >= len(arr) {
		return nil
	}
	return arr[i]
}
