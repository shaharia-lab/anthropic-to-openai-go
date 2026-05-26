package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Content part types for multimodal message content.
const (
	PartTypeText     = "text"
	PartTypeImageURL = "image_url"
)

// jsonNull is the literal JSON null token.
var jsonNull = []byte("null")

// Content is a chat message body. The OpenAI API encodes it either as a plain
// string or as an array of typed parts (text and images). After unmarshalling,
// Parts is populated when the wire value was an array; otherwise Text holds the
// scalar string.
type Content struct {
	Text  string
	Parts []ContentPart
}

// ContentPart is a single element of multimodal message content.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL references an image by remote URL or inline RFC 2397 data URL.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// IsZero reports whether the content carries neither text nor parts.
func (c Content) IsZero() bool {
	return c.Text == "" && len(c.Parts) == 0
}

// UnmarshalJSON decodes either a JSON string or an array of content parts.
func (c *Content) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, jsonNull) {
		return nil
	}
	switch trimmed[0] {
	case '"':
		return json.Unmarshal(trimmed, &c.Text)
	case '[':
		return json.Unmarshal(trimmed, &c.Parts)
	default:
		return fmt.Errorf("openai: message content must be a string or array, got %q", trimmed[0])
	}
}

// MarshalJSON encodes parts as an array when present, otherwise the text as a
// JSON string.
func (c Content) MarshalJSON() ([]byte, error) {
	if len(c.Parts) > 0 {
		return json.Marshal(c.Parts)
	}
	return json.Marshal(c.Text)
}
