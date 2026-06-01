package gemini

// DefaultBuildLabel is the fallback Gemini web build label used when auto-resolution
// is disabled or fails. It is date-stamped and may go stale.
const DefaultBuildLabel = "boq_assistant-bard-web-server_20260529.02_p0"

// Defaults applied by New to any Config field left at its zero value, so a
// consumer can pass gemini.Config{} (optionally with just a cookie) and get a
// working client.
const (
	// DefaultModel is used when Config.DefaultModel is empty.
	DefaultModel = "gemini-3.5-flash"
	// DefaultRequestTimeout is the per-request HTTP timeout (seconds) used when
	// Config.RequestTimeout is zero.
	DefaultRequestTimeout = 180
	// DefaultRetryAttempts is the blocking-generate retry count used when
	// Config.RetryAttempts is zero. It must be at least 1 or Generate makes no
	// request at all.
	DefaultRetryAttempts = 3
	// DefaultRetryDelaySec is the delay between retries (seconds) used when
	// Config.RetryDelaySec is zero.
	DefaultRetryDelaySec = 2
)

// Config holds the upstream settings the Gemini web client needs. A consumer
// populates it directly; the proxy's main package builds it from environment
// variables. All fields are plain values with no smart defaults except GeminiBL,
// which falls back to DefaultBuildLabel inside New when left empty.
type Config struct {
	Proxy          string // optional HTTP/HTTPS proxy URL
	RequestTimeout int    // per-request timeout, seconds
	GeminiBL       string // build label (bl) sent to the backend
	GeminiBLAuto   bool   // auto-resolve the build label from the Gemini page (defaults on when GeminiBL is unset)
	AuthUser       string // /u/N account index for non-default Google accounts
	DefaultModel   string // model used when a request names none
	Cookie         string // inline cookie string
	CookieFile     string // path to a cookie file (raw "k=v; …" string)
	RetryAttempts  int    // blocking-generate retry attempts
	RetryDelaySec  int    // delay between retries, seconds
}

// applyDefaults fills any field left at its zero value with the corresponding
// Default* constant, so gemini.Config{} produces a usable client. When no explicit
// GeminiBL is given it also turns GeminiBLAuto on, so the date-stamped fallback
// label is kept current by auto-resolution rather than going stale; setting an
// explicit GeminiBL pins it and leaves auto-resolution off.
func (c *Config) applyDefaults() {
	if c.GeminiBL == "" {
		c.GeminiBL = DefaultBuildLabel
		c.GeminiBLAuto = true
	}
	// The numeric timing/retry fields treat any value below 1 as "unset": 0 is the
	// zero-value sentinel, and a negative value is a misconfiguration that would
	// otherwise misbehave — a negative RequestTimeout makes http.Client time out
	// immediately, and a negative RetryAttempts makes Generate's loop run zero times
	// and wrap a nil error. Clamp all of them to their defaults.
	if c.RequestTimeout < 1 {
		c.RequestTimeout = DefaultRequestTimeout
	}
	if c.RetryAttempts < 1 {
		c.RetryAttempts = DefaultRetryAttempts
	}
	if c.RetryDelaySec < 1 {
		c.RetryDelaySec = DefaultRetryDelaySec
	}
	if c.DefaultModel == "" {
		c.DefaultModel = DefaultModel
	}
}
