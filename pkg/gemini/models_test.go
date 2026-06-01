package gemini

import (
	"encoding/json"
	"testing"
)

// testModels is the deterministic registry used by tests so they never hit the
// network. gemini-3.5-flash matches testConfig().DefaultModel.
func testModels() []*AvailableModel {
	return []*AvailableModel{
		{Name: "gemini-3.5-flash", ModelID: "fastid", DisplayName: "Fast", Description: "Fast general-purpose model", Capacity: 1, CapacityField: 13},
		{Name: "gemini-3.1-pro", ModelID: "proid", DisplayName: "Pro", Description: "Pro model", Capacity: 2, CapacityField: 12},
		{Name: "gemini-3.1-flash-lite", ModelID: "liteid", DisplayName: "3.1 Flash-Lite", Description: "Lightweight fast model", Capacity: 1, CapacityField: 12},
	}
}

// seedModels installs a registry on the client, bypassing the network. With no
// models it installs the default test set.
func seedModels(c *Client, models ...*AvailableModel) {
	if len(models) == 0 {
		models = testModels()
	}
	c.storeModels(models, "test")
}

func TestComputeCapacity(t *testing.T) {
	cases := []struct {
		name      string
		tier, cap []float64
		capacity  int
		field     int
	}{
		{"tier21", []float64{21}, nil, 1, 13},
		{"tier22", []float64{22}, nil, 2, 13},
		{"plus115", nil, []float64{115}, 4, 12},
		{"tier16", []float64{16}, nil, 3, 12},
		{"cap106", nil, []float64{106}, 3, 12},
		{"tier8", []float64{8}, nil, 2, 12},
		{"cap19", nil, []float64{19}, 2, 12},
		{"cap19+106", nil, []float64{19, 106}, 3, 12},
		{"free", nil, nil, 1, 12},
		{"tier21_wins", []float64{21, 22, 16}, []float64{115}, 1, 13},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cap, field := computeCapacity(tc.tier, tc.cap)
			if cap != tc.capacity || field != tc.field {
				t.Errorf("computeCapacity(%v,%v) = (%d,%d), want (%d,%d)", tc.tier, tc.cap, cap, field, tc.capacity, tc.field)
			}
		})
	}
}

func TestSlugifyModelName(t *testing.T) {
	cases := map[string]string{
		"3.5 Flash":       "gemini-3.5-flash",
		"3.1 Pro":         "gemini-3.1-pro",
		"3.1 Flash-Lite":  "gemini-3.1-flash-lite",
		"Fast":            "gemini-fast",
		"  Multi   Word ": "gemini-multi-word",
	}
	for in, want := range cases {
		if got := slugifyModelName(in); got != want {
			t.Errorf("slugifyModelName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestModelSlug(t *testing.T) {
	// Versioned name [11] wins.
	if got := modelSlug("3.1 Flash-Lite", "Fast", "hexid"); got != "gemini-3.1-flash-lite" {
		t.Errorf("versioned = %q", got)
	}
	// Fallback to the short label [1].
	if got := modelSlug("", "Fast", "hexid"); got != "gemini-fast" {
		t.Errorf("label fallback = %q", got)
	}
	// Fallback to gemini-<id>.
	if got := modelSlug("", "", "hexid"); got != "gemini-hexid" {
		t.Errorf("id fallback = %q", got)
	}
}

func TestSelectionHeader(t *testing.T) {
	t.Run("field 12", func(t *testing.T) {
		m := &AvailableModel{ModelID: "abc123", Capacity: 2, CapacityField: 12}
		want := `[1,null,null,null,"abc123",null,null,0,[4],null,null,2]`
		if got := m.selectionHeader(); got != want {
			t.Errorf("selectionHeader = %q, want %q", got, want)
		}
	})
	t.Run("field 13", func(t *testing.T) {
		m := &AvailableModel{ModelID: "abc123", Capacity: 2, CapacityField: 13}
		want := `[1,null,null,null,"abc123",null,null,0,[4],null,null,null,2]`
		if got := m.selectionHeader(); got != want {
			t.Errorf("selectionHeader = %q, want %q", got, want)
		}
	})
}

func TestModelHeaders(t *testing.T) {
	m := &AvailableModel{ModelID: "abc123", Capacity: 1, CapacityField: 12}
	h := m.headers("UUID-1")
	if h["x-goog-ext-525001261-jspb"] != m.selectionHeader() {
		t.Errorf("selection header = %q", h["x-goog-ext-525001261-jspb"])
	}
	if h["x-goog-ext-73010989-jspb"] != "[0]" || h["x-goog-ext-73010990-jspb"] != "[0]" {
		t.Errorf("zero headers = %q / %q", h["x-goog-ext-73010989-jspb"], h["x-goog-ext-73010990-jspb"])
	}
	if h["x-goog-ext-525005358-jspb"] != `["UUID-1",1]` {
		t.Errorf("uuid header = %q, want [\"UUID-1\",1]", h["x-goog-ext-525005358-jspb"])
	}
}

func TestParseBatchResponse(t *testing.T) {
	inner := `[null,"payload"]`
	top := []any{[]any{"wrb.fr", "otAQ7b", inner, nil, nil, nil, "generic"}}
	raw, err := json.Marshal(top)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// Mirror the real response: )]}' guard, a length token line, then the JSON.
	full := ")]}'\n\n123\n" + string(raw) + "\n"

	got, ok := parseBatchResponse(full, "otAQ7b")
	if !ok || got != inner {
		t.Errorf("parseBatchResponse = (%q, %v), want (%q, true)", got, ok, inner)
	}

	if _, ok := parseBatchResponse(full, "other"); ok {
		t.Errorf("expected miss for a different rpcid")
	}
}

func TestParseModels(t *testing.T) {
	entry := func(id, label, desc, versioned string) []any {
		m := make([]any, 12)
		m[0] = id
		m[1] = label
		m[2] = desc
		m[11] = versioned
		return m
	}
	body := make([]any, 18)
	body[15] = []any{
		entry("hexfast", "Fast", "fast model", "3.5 Flash"),
		entry("hexpro", "Pro", "pro model", "3.1 Pro"),
		entry("hexlabel", "Quick", "", ""), // no versioned name → label slug
	}
	body[16] = []any{} // empty tier flags → Free (1, 12)
	body[17] = []any{}
	innerJSON, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	models, err := parseModels(string(innerJSON))
	if err != nil {
		t.Fatalf("parseModels: %v", err)
	}
	if len(models) != 3 {
		t.Fatalf("got %d models, want 3", len(models))
	}
	want := []struct{ name, id string }{
		{"gemini-3.5-flash", "hexfast"},
		{"gemini-3.1-pro", "hexpro"},
		{"gemini-quick", "hexlabel"},
	}
	for i, w := range want {
		if models[i].Name != w.name || models[i].ModelID != w.id {
			t.Errorf("model[%d] = (%q, %q), want (%q, %q)", i, models[i].Name, models[i].ModelID, w.name, w.id)
		}
		if models[i].Capacity != 1 || models[i].CapacityField != 12 {
			t.Errorf("model[%d] capacity = (%d,%d), want (1,12)", i, models[i].Capacity, models[i].CapacityField)
		}
	}

	if _, err := parseModels(`[]`); err == nil {
		t.Errorf("expected error for a body with no models")
	}
}

func TestFallbackModels(t *testing.T) {
	// The fallback catalog is exactly the configured default (no fictitious
	// models that would silently route to Flash).
	models := fallbackModels("gemini-3.5-flash")
	if len(models) != 1 || models[0].Name != "gemini-3.5-flash" {
		t.Errorf("fallback = %+v, want a single gemini-3.5-flash entry", models)
	}
	// A custom default is used as the sole entry.
	custom := fallbackModels("gemini-custom")
	if len(custom) != 1 || custom[0].Name != "gemini-custom" {
		t.Errorf("custom default = %+v, want a single gemini-custom entry", custom)
	}
	// An empty default falls back to gemini-3.5-flash.
	empty := fallbackModels("")
	if len(empty) != 1 || empty[0].Name != "gemini-3.5-flash" {
		t.Errorf("empty default = %+v, want a single gemini-3.5-flash entry", empty)
	}
}

func TestResolveModelRegistry(t *testing.T) {
	c, _ := New(testConfig(), testLogger())
	seedModels(c)

	if m, err := c.ResolveModel("gemini-3.1-pro"); err != nil || m.Name != "gemini-3.1-pro" {
		t.Errorf("known = (%v, %v)", m, err)
	}
	if m, err := c.ResolveModel(""); err != nil || m.Name != "gemini-3.5-flash" {
		t.Errorf("default = (%v, %v)", m, err)
	}
	if _, err := c.ResolveModel("nope"); err == nil {
		t.Errorf("expected error for unknown model")
	}
	// Foreign id falls back to the default.
	if m := c.ResolveModelOrDefault("claude-3"); m.Name != "gemini-3.5-flash" {
		t.Errorf("orDefault = %q, want gemini-3.5-flash", m.Name)
	}
}

func TestDefaultModelFallsBackToFirst(t *testing.T) {
	cfg := testConfig()
	cfg.DefaultModel = "gemini-absent"
	c, _ := New(cfg, testLogger())
	seedModels(c)
	// The configured default is absent, so the first registered model is used.
	if m := c.defaultModel(); m == nil || m.Name != "gemini-3.5-flash" {
		t.Errorf("defaultModel = %v, want first (gemini-3.5-flash)", m)
	}
}

func TestSetModels(t *testing.T) {
	c, _ := New(testConfig(), testLogger())
	c.SetModels(testModels())
	if got := c.ModelsSourceLabel(); got != "static" {
		t.Errorf("source = %q, want static", got)
	}
	names := c.ModelNames()
	if len(names) != 3 || names[0] != "gemini-3.5-flash" {
		t.Errorf("names = %v", names)
	}
}
