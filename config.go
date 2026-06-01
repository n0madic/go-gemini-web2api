package main

import (
	"bufio"
	"os"
	"strconv"
	"strings"

	"github.com/n0madic/go-gemini-web2api/pkg/gemini"
)

const version = "0.1.0"

// defaultListen is the default bind address (localhost only) used when GEMINI_LISTEN
// is unset.
const defaultListen = "127.0.0.1:8081"

// Config holds all runtime settings. It is populated entirely from environment
// variables (optionally seeded from a .env file); there is no JSON config file.
// The upstream Gemini settings live in the embedded gemini.Config; the rest are
// the proxy's own network/CLI concerns.
type Config struct {
	Listen        string   // bind address, e.g. 127.0.0.1:8081 or :8081
	LogRequests   bool     // log each request at Info level
	CookieRefresh int      // minutes between __Secure-1PSIDTS rotations (0 = off)
	APIKeys       []string // allowed client keys; empty means open access
	Gemini        gemini.Config
}

// loadConfig seeds env from a .env file, then reads all settings. The .env path
// is resolved in priority order: the explicit envFile argument (the -config
// flag), then $GEMINI_ENV_FILE, then ./.env.
func loadConfig(envFile string) *Config {
	if envFile == "" {
		envFile = os.Getenv("GEMINI_ENV_FILE")
	}
	if envFile == "" {
		envFile = ".env"
	}
	loadDotenv(envFile)

	cfg := &Config{
		Listen:        normalizeListen(getEnv("GEMINI_LISTEN", defaultListen)),
		LogRequests:   getEnvBool("GEMINI_LOG_REQUESTS", false),
		CookieRefresh: getEnvInt("GEMINI_COOKIE_REFRESH_MIN", 9),
		APIKeys:       splitKeys(getEnv("GEMINI_API_KEYS", "")),
		Gemini: gemini.Config{
			RetryAttempts:  getEnvInt("GEMINI_RETRY_ATTEMPTS", 3),
			RetryDelaySec:  getEnvInt("GEMINI_RETRY_DELAY_SEC", 2),
			RequestTimeout: getEnvInt("GEMINI_REQUEST_TIMEOUT_SEC", 180),
			AuthUser:       getEnv("GEMINI_AUTH_USER", ""),
			DefaultModel:   getEnv("GEMINI_DEFAULT_MODEL", "gemini-3.5-flash"),
			Cookie:         getEnv("GEMINI_COOKIE", ""),
			CookieFile:     getEnv("GEMINI_COOKIE_FILE", ""),
			Proxy:          getEnv("GEMINI_PROXY", ""),
		},
	}

	// GEMINI_BL: empty or "auto" → auto-resolve from the Gemini page (with the
	// baked-in default as fallback); an explicit value disables auto-resolution.
	bl := strings.TrimSpace(getEnv("GEMINI_BL", ""))
	cfg.Gemini.GeminiBLAuto = bl == "" || strings.EqualFold(bl, "auto")
	if cfg.Gemini.GeminiBLAuto {
		cfg.Gemini.GeminiBL = gemini.DefaultBuildLabel
	} else {
		cfg.Gemini.GeminiBL = bl
	}
	return cfg
}

// normalizeListen turns a listen value into a "host:port" bind address. It accepts
// a full "host:port", a port-only ":port", or a bare numeric port (treated as
// ":port", i.e. all interfaces). An empty value yields the default.
func normalizeListen(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return defaultListen
	}
	// A bare numeric port ("8081") is shorthand for ":8081".
	if !strings.Contains(s, ":") {
		if _, err := strconv.Atoi(s); err == nil {
			return ":" + s
		}
	}
	return s
}

// loadDotenv parses a simple KEY=VALUE file and sets variables that are not
// already present in the process environment (standard dotenv precedence).
func loadDotenv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return // missing .env is not an error
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		val = unquote(val)
		if key == "" {
			continue
		}
		if _, exists := os.LookupEnv(key); !exists {
			_ = os.Setenv(key, val)
		}
	}
}

// unquote removes a single matching pair of surrounding single or double quotes.
func unquote(s string) string {
	if len(s) >= 2 {
		if (s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'') {
			return s[1 : len(s)-1]
		}
	}
	return s
}

func getEnv(key, def string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return def
}

func getEnvInt(key string, def int) int {
	if v, ok := os.LookupEnv(key); ok {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
			return n
		}
	}
	return def
}

func getEnvBool(key string, def bool) bool {
	if v, ok := os.LookupEnv(key); ok {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "1", "true", "yes", "on":
			return true
		case "0", "false", "no", "off":
			return false
		}
	}
	return def
}

// splitKeys parses a comma-separated list into a trimmed, non-empty slice.
func splitKeys(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
