package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestIsLoopbackBind(t *testing.T) {
	cases := map[string]bool{
		"127.0.0.1:8081":  true,  // explicit loopback
		"localhost:8081":  true,  // loopback hostname
		"[::1]:8081":      true,  // IPv6 loopback
		":8081":           false, // wildcard (all interfaces)
		"0.0.0.0:9000":    false, // explicit wildcard
		"192.168.1.5:443": false, // LAN address
		"[::]:8081":       false, // IPv6 wildcard
		"garbage":         false, // unparseable → treated as non-loopback (safe default)
	}
	for addr, want := range cases {
		if got := isLoopbackBind(addr); got != want {
			t.Errorf("isLoopbackBind(%q) = %v, want %v", addr, got, want)
		}
	}
}

func TestNewHTTPServerTimeouts(t *testing.T) {
	cfg := &Config{Listen: "127.0.0.1:0"}
	srv := newHTTPServer(cfg, http.NewServeMux())

	if srv.Addr != cfg.Listen {
		t.Errorf("Addr = %q, want %q", srv.Addr, cfg.Listen)
	}
	if srv.ReadHeaderTimeout != httpReadHeaderTimeout {
		t.Errorf("ReadHeaderTimeout = %v, want %v", srv.ReadHeaderTimeout, httpReadHeaderTimeout)
	}
	if srv.ReadTimeout != httpReadTimeout {
		t.Errorf("ReadTimeout = %v, want %v", srv.ReadTimeout, httpReadTimeout)
	}
	if srv.IdleTimeout != httpIdleTimeout {
		t.Errorf("IdleTimeout = %v, want %v", srv.IdleTimeout, httpIdleTimeout)
	}
	// WriteTimeout must stay unset so SSE streams can stay open for the full
	// generation; a non-zero value would cut long streaming responses short.
	if srv.WriteTimeout != 0 {
		t.Errorf("WriteTimeout = %v, want 0 (unset for SSE)", srv.WriteTimeout)
	}
	if srv.Handler == nil {
		t.Error("Handler must not be nil")
	}
}

func TestReadBodyTooLargeReturns413(t *testing.T) {
	s := newTestServer(t, testConfig())
	// A body just over the cap must be rejected with an explicit 413, not silently
	// truncated into a confusing JSON parse error (400).
	big := strings.Repeat("a", maxBodyBytes+1)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(big))
	rr := httptest.NewRecorder()
	s.handler().ServeHTTP(rr, req)
	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("status = %d, want 413", rr.Code)
	}
}
