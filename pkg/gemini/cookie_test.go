package gemini

import "testing"

func TestReplaceCookieValue(t *testing.T) {
	in := "__Secure-1PSIDTS=old; __Secure-1PSID=sid"
	got := replaceCookieValue(in, "__Secure-1PSIDTS", "new")
	if got != "__Secure-1PSIDTS=new; __Secure-1PSID=sid" {
		t.Errorf("replace = %q", got)
	}

	// Key absent → appended.
	got = replaceCookieValue("__Secure-1PSID=sid", "__Secure-1PSIDTS", "new")
	if got != "__Secure-1PSID=sid; __Secure-1PSIDTS=new" {
		t.Errorf("append = %q", got)
	}

	// Other values untouched, formatting normalized.
	got = replaceCookieValue("A=1;B=2;__Secure-1PSIDTS=x", "__Secure-1PSIDTS", "y")
	if got != "A=1; B=2; __Secure-1PSIDTS=y" {
		t.Errorf("mixed = %q", got)
	}
}

func TestCookieValue(t *testing.T) {
	jar := "FOO=bar; SAPISID=secret123; __Secure-1PSIDTS=tsval"
	if got := cookieValue(jar, "SAPISID"); got != "secret123" {
		t.Errorf("SAPISID = %q, want secret123", got)
	}
	if got := cookieValue(jar, "__Secure-1PSIDTS"); got != "tsval" {
		t.Errorf("__Secure-1PSIDTS = %q, want tsval", got)
	}
	// Tolerates a ";" separator without a following space.
	if got := cookieValue("A=1;SAPISID=s;B=2", "SAPISID"); got != "s" {
		t.Errorf("no-space SAPISID = %q, want s", got)
	}
	if got := cookieValue("NO=match", "SAPISID"); got != "" {
		t.Errorf("absent = %q, want empty", got)
	}
}

func TestSetCookieValue(t *testing.T) {
	v, ok := setCookieValue("__Secure-1PSIDTS=abc123; Path=/; Secure; HttpOnly", "__Secure-1PSIDTS")
	if !ok || v != "abc123" {
		t.Errorf("got (%q, %v)", v, ok)
	}
	if _, ok := setCookieValue("NID=xyz; Path=/", "__Secure-1PSIDTS"); ok {
		t.Errorf("should not match a different cookie")
	}
}
