package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/n0madic/go-gemini-web2api/pkg/gemini"
)

func TestUnquote(t *testing.T) {
	cases := map[string]string{
		`"hello"`: "hello",
		`'world'`: "world",
		`plain`:   "plain",
		`"mix'`:   `"mix'`,
		`""`:      "",
	}
	for in, want := range cases {
		if got := unquote(in); got != want {
			t.Errorf("unquote(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestGetEnvBool(t *testing.T) {
	cases := []struct {
		val  string
		want bool
	}{
		{"true", true}, {"1", true}, {"yes", true}, {"on", true},
		{"false", false}, {"0", false}, {"no", false}, {"off", false},
	}
	for _, tc := range cases {
		t.Setenv("GEMINI_TEST_BOOL", tc.val)
		if got := getEnvBool("GEMINI_TEST_BOOL", !tc.want); got != tc.want {
			t.Errorf("getEnvBool(%q) = %v, want %v", tc.val, got, tc.want)
		}
	}
	if got := getEnvBool("GEMINI_TEST_BOOL_MISSING", true); !got {
		t.Errorf("missing key should return default")
	}
}

func TestGetEnvInt(t *testing.T) {
	t.Setenv("GEMINI_TEST_INT", "42")
	if got := getEnvInt("GEMINI_TEST_INT", 7); got != 42 {
		t.Errorf("getEnvInt = %d, want 42", got)
	}
	t.Setenv("GEMINI_TEST_INT", "notanumber")
	if got := getEnvInt("GEMINI_TEST_INT", 7); got != 7 {
		t.Errorf("getEnvInt with bad value = %d, want default 7", got)
	}
}

func TestSplitKeys(t *testing.T) {
	got := splitKeys("a, b ,, c")
	want := []string{"a", "b", "c"}
	if len(got) != len(want) {
		t.Fatalf("splitKeys = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("splitKeys[%d] = %q, want %q", i, got[i], want[i])
		}
	}
	if splitKeys("  ") != nil {
		t.Errorf("blank input should return nil")
	}
}

func TestLoadDotenv(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".env")
	content := "" +
		"# a comment\n" +
		"\n" +
		"GEMINI_TEST_DOTENV_A=value_a\n" +
		"export GEMINI_TEST_DOTENV_B=\"quoted value\"\n" +
		"GEMINI_TEST_DOTENV_C='single'\n" +
		"GEMINI_TEST_DOTENV_PRESET=fromfile\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write .env: %v", err)
	}

	// Pre-set one var to verify dotenv does NOT override existing env.
	t.Setenv("GEMINI_TEST_DOTENV_PRESET", "fromenv")

	for _, k := range []string{"GEMINI_TEST_DOTENV_A", "GEMINI_TEST_DOTENV_B", "GEMINI_TEST_DOTENV_C"} {
		os.Unsetenv(k)
		t.Cleanup(func() { os.Unsetenv(k) })
	}

	loadDotenv(path)

	checks := map[string]string{
		"GEMINI_TEST_DOTENV_A":      "value_a",
		"GEMINI_TEST_DOTENV_B":      "quoted value",
		"GEMINI_TEST_DOTENV_C":      "single",
		"GEMINI_TEST_DOTENV_PRESET": "fromenv", // not overridden
	}
	for k, want := range checks {
		if got := os.Getenv(k); got != want {
			t.Errorf("%s = %q, want %q", k, got, want)
		}
	}
}

func TestLoadDotenvMissingFile(t *testing.T) {
	// Should not panic or error on a missing file.
	loadDotenv(filepath.Join(t.TempDir(), "does-not-exist.env"))
}

func TestLoadConfigBuildLabelAuto(t *testing.T) {
	noEnv := filepath.Join(t.TempDir(), "absent.env")
	prev, had := os.LookupEnv("GEMINI_BL")
	t.Cleanup(func() {
		if had {
			os.Setenv("GEMINI_BL", prev)
		} else {
			os.Unsetenv("GEMINI_BL")
		}
	})

	os.Unsetenv("GEMINI_BL")
	if cfg := loadConfig(noEnv); !cfg.Gemini.GeminiBLAuto || cfg.Gemini.GeminiBL != gemini.DefaultBuildLabel {
		t.Errorf("unset: auto=%v bl=%q", cfg.Gemini.GeminiBLAuto, cfg.Gemini.GeminiBL)
	}

	os.Setenv("GEMINI_BL", "auto")
	if cfg := loadConfig(noEnv); !cfg.Gemini.GeminiBLAuto || cfg.Gemini.GeminiBL != gemini.DefaultBuildLabel {
		t.Errorf("auto: auto=%v bl=%q", cfg.Gemini.GeminiBLAuto, cfg.Gemini.GeminiBL)
	}

	os.Setenv("GEMINI_BL", "boq_custom_123")
	if cfg := loadConfig(noEnv); cfg.Gemini.GeminiBLAuto || cfg.Gemini.GeminiBL != "boq_custom_123" {
		t.Errorf("explicit: auto=%v bl=%q", cfg.Gemini.GeminiBLAuto, cfg.Gemini.GeminiBL)
	}
}

func TestLoadConfigUsesConfigPath(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "custom.env")
	content := "GEMINI_LISTEN=0.0.0.0:9099\nGEMINI_DEFAULT_MODEL=gemini-3.1-pro\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write env: %v", err)
	}

	// Ensure the keys are unset so the file values apply, restoring afterwards.
	for _, k := range []string{"GEMINI_LISTEN", "GEMINI_DEFAULT_MODEL"} {
		prev, had := os.LookupEnv(k)
		os.Unsetenv(k)
		t.Cleanup(func() {
			if had {
				os.Setenv(k, prev)
			} else {
				os.Unsetenv(k)
			}
		})
	}

	cfg := loadConfig(path)
	if cfg.Listen != "0.0.0.0:9099" {
		t.Errorf("Listen = %q, want 0.0.0.0:9099", cfg.Listen)
	}
	if cfg.Gemini.DefaultModel != "gemini-3.1-pro" {
		t.Errorf("DefaultModel = %q, want gemini-3.1-pro", cfg.Gemini.DefaultModel)
	}
}

func TestNormalizeListen(t *testing.T) {
	cases := map[string]string{
		"":                "127.0.0.1:8081", // default
		"  ":              "127.0.0.1:8081", // whitespace → default
		"127.0.0.1:8081":  "127.0.0.1:8081", // full host:port
		":8081":           ":8081",          // port-only → all interfaces
		"8081":            ":8081",          // bare port → :port
		"0.0.0.0:9000":    "0.0.0.0:9000",   // explicit wildcard
		"192.168.1.5:443": "192.168.1.5:443",
	}
	for in, want := range cases {
		if got := normalizeListen(in); got != want {
			t.Errorf("normalizeListen(%q) = %q, want %q", in, got, want)
		}
	}
}
