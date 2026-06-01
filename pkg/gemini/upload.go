package gemini

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
)

// uploadEndpoint is Google's push (Scotty) endpoint for image attachments. It
// accepts an unauthenticated multipart POST and returns an opaque file reference.
const uploadEndpoint = "https://content-push.googleapis.com/upload/"

// uploadPushID is the fixed push channel id the Gemini web client uses for uploads.
const uploadPushID = "feeds/mcudyrk2a4khkz"

// maxImageBytes caps both a fetched remote image and an upload response read.
const maxImageBytes = 20 << 20

// uploadedImage is a successfully uploaded attachment: its push-service reference
// and the filename presented to the model.
type uploadedImage struct {
	ref      string
	filename string
}

// imageRefsField builds the inner[0][3] attachment list as [[[ref], filename], …],
// or nil when there are no images (so JSON serializes the slot as null).
func imageRefsField(images []uploadedImage) any {
	if len(images) == 0 {
		return nil
	}
	out := make([]any, len(images))
	for i, im := range images {
		out[i] = []any{[]any{im.ref}, im.filename}
	}
	return out
}

// uploadImages resolves and uploads each input image, returning the successful
// uploads. A failed image (bad data, fetch error, upload error) is logged and
// skipped so the request still proceeds with the remaining images and text.
func (c *Client) uploadImages(ctx context.Context, imgs []InputImage) []uploadedImage {
	if len(imgs) == 0 {
		return nil
	}
	out := make([]uploadedImage, 0, len(imgs))
	for i := range imgs {
		data, err := c.imageBytes(ctx, imgs[i])
		if err != nil {
			c.log.Warn("skipping image: cannot obtain bytes", "index", i, "err", err)
			continue
		}
		name := imgs[i].Filename
		if name == "" {
			name = "image.png"
		}
		ref, err := c.uploadImage(ctx, data, name)
		if err != nil {
			c.log.Warn("skipping image: upload failed", "index", i, "err", err)
			continue
		}
		out = append(out, uploadedImage{ref: ref, filename: name})
	}
	return out
}

// imageBytes returns the image's bytes, fetching a remote URL when needed.
func (c *Client) imageBytes(ctx context.Context, img InputImage) ([]byte, error) {
	if len(img.Data) > 0 {
		return img.Data, nil
	}
	if img.URL == "" {
		return nil, errors.New("image has neither inline data nor URL")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, img.URL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetch image: http %d", resp.StatusCode)
	}
	return io.ReadAll(io.LimitReader(resp.Body, maxImageBytes))
}

// uploadImage POSTs the image to the push endpoint (multipart, no auth) and
// returns the resulting file reference string.
func (c *Client) uploadImage(ctx context.Context, data []byte, filename string) (string, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	part, err := mw.CreateFormFile("file", filename)
	if err != nil {
		return "", err
	}
	if _, err := part.Write(data); err != nil {
		return "", err
	}
	if err := mw.Close(); err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, uploadEndpoint, &buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("Push-ID", uploadPushID)

	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, maxImageBytes))
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("upload http %d", resp.StatusCode)
	}
	ref := strings.TrimSpace(string(body))
	if ref == "" {
		return "", errors.New("empty upload reference")
	}
	return ref, nil
}
