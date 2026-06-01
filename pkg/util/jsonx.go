// Package util provides utility functions for JSON encoding and decoding.
package util

import (
	"bytes"
	"encoding/json"
)

// MarshalNoEscape serializes v without escaping <, >, & so text is emitted
// verbatim, and trims the trailing newline json.Encoder appends.
func MarshalNoEscape(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}
