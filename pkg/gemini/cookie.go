package gemini

import (
	"context"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// rotateEndpoint issues a fresh __Secure-1PSIDTS via Set-Cookie.
const rotateEndpoint = "https://accounts.google.com/RotateCookies"

// rotatePayload is the fixed body the Gemini web client sends to RotateCookies.
const rotatePayload = `[000,"-0000000000000000000"]`

// Google cookie names used by the proxy: SAPISID feeds the SAPISIDHASH auth
// header; __Secure-1PSIDTS is the rotating session-freshness token.
const (
	cookieSAPISID = "SAPISID"
	cookiePSIDTS  = "__Secure-1PSIDTS"
)

// CookieRefreshLoop periodically rotates __Secure-1PSIDTS so the configured cookie
// (file-backed or inline) does not go stale, and refreshes the model list after
// each rotation (a renewed session may surface a changed account catalog). The
// initial rotation is expected to be done synchronously at startup, so the loop
// only ticks on the interval and stops when ctx is cancelled.
func (c *Client) CookieRefreshLoop(ctx context.Context, interval time.Duration) {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			rctx, cancel := context.WithTimeout(ctx, 30*time.Second)
			c.RotateCookie(rctx)
			c.fetchModels(rctx)
			cancel()
		}
	}
}

// RotateCookie fetches a fresh __Secure-1PSIDTS and applies it. A file-backed
// cookie is rewritten in place (so it survives restarts); an inline cookie is
// rotated in memory only (it survives the process run but not a restart).
func (c *Client) RotateCookie(ctx context.Context) {
	cookie, sapisid := c.loadCookie()
	if cookie == "" || cookieValue(cookie, cookiePSIDTS) == "" {
		return
	}
	newTS, err := c.fetchRotatedTS(ctx, cookie)
	if err != nil {
		c.log.Warn("cookie rotation request failed", "err", err)
		return
	}
	if newTS == "" {
		return // Google issued no new token this cycle
	}
	newCookie := replaceCookieValue(cookie, cookiePSIDTS, newTS)

	// Persist to disk only for a file-backed cookie; an inline cookie has no file.
	if c.cfg.Cookie == "" && c.cfg.CookieFile != "" {
		if err := os.WriteFile(c.cfg.CookieFile, []byte(newCookie), 0o600); err != nil {
			c.log.Warn("cookie rotation persist failed", "err", err)
			return
		}
	}

	c.cookieMu.Lock()
	c.cookieStr, c.cookieSAPI, c.cookieLoaded = newCookie, sapisid, true
	if c.cfg.CookieFile != "" {
		if info, err := os.Stat(c.cfg.CookieFile); err == nil {
			c.cookieMtime = info.ModTime()
		}
	}
	c.cookieMu.Unlock()
	c.log.Info("rotated __Secure-1PSIDTS cookie")
	// Keep the XSRF "at" token in sync with the rotated session. The model list is
	// refreshed by the caller (startup, then the periodic refresh loop).
	c.refreshAtToken(ctx)
}

// fetchRotatedTS POSTs to RotateCookies and returns the fresh __Secure-1PSIDTS
// value from the response Set-Cookie header (empty if none was issued).
func (c *Client) fetchRotatedTS(ctx context.Context, cookie string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, rotateEndpoint, strings.NewReader(rotatePayload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("Cookie", cookie)

	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)

	for _, sc := range resp.Header["Set-Cookie"] {
		if v, ok := setCookieValue(sc, cookiePSIDTS); ok {
			return v, nil
		}
	}
	return "", nil
}

// cookieValue returns the value of the named cookie in a "k=v; k=v" cookie jar
// string, or "" when absent. It tolerates either "; " or ";" between pairs.
func cookieValue(cookie, name string) string {
	for _, pair := range strings.Split(cookie, ";") {
		if k, v, ok := strings.Cut(strings.TrimSpace(pair), "="); ok && k == name {
			return v
		}
	}
	return ""
}

// setCookieValue extracts the value of key from a Set-Cookie header line.
func setCookieValue(setCookie, key string) (string, bool) {
	if !strings.HasPrefix(setCookie, key+"=") {
		return "", false
	}
	v := setCookie[len(key)+1:]
	if i := strings.IndexByte(v, ';'); i >= 0 {
		v = v[:i]
	}
	return v, true
}

// replaceCookieValue returns cookieStr with key's value replaced (appending the
// pair if key is absent), normalizing the "k=v; k=v" formatting.
func replaceCookieValue(cookieStr, key, value string) string {
	var out []string
	found := false
	for _, p := range strings.Split(cookieStr, ";") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		k, _, _ := strings.Cut(p, "=")
		if k == key {
			out = append(out, key+"="+value)
			found = true
		} else {
			out = append(out, p)
		}
	}
	if !found {
		out = append(out, key+"="+value)
	}
	return strings.Join(out, "; ")
}
