package gemini

import "testing"

func TestImageRefsField(t *testing.T) {
	if got := imageRefsField(nil); got != nil {
		t.Errorf("imageRefsField(nil) = %v, want nil", got)
	}
	out := imageRefsField([]uploadedImage{{ref: "R", filename: "f.png"}})
	arr, ok := out.([]any)
	if !ok || len(arr) != 1 {
		t.Fatalf("imageRefsField = %v", out)
	}
	entry := arr[0].([]any)
	if rw := entry[0].([]any); rw[0] != "R" {
		t.Errorf("ref wrap = %v", entry[0])
	}
	if entry[1] != "f.png" {
		t.Errorf("filename = %v", entry[1])
	}
}
