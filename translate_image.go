package anthropic2openai

import (
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/shaharia-lab/anthropic-to-openai-go/anthropic"
)

// dataURLPrefix is the scheme prefix of an RFC 2397 data URL.
const dataURLPrefix = "data:"

// base64Token is the data-URL parameter marking base64-encoded payloads.
const base64Token = "base64"

// imageBlock builds an Anthropic image content block from an OpenAI image URL.
// The URL may be a remote http(s) reference or an inline base64 data URL.
func imageBlock(url string) (anthropic.ContentBlock, error) {
	if url == "" {
		return anthropic.ContentBlock{}, fmt.Errorf("image_url is empty")
	}
	if strings.HasPrefix(url, dataURLPrefix) {
		source, err := parseDataURL(url)
		if err != nil {
			return anthropic.ContentBlock{}, err
		}
		return anthropic.ContentBlock{Type: anthropic.BlockImage, Source: source}, nil
	}
	return anthropic.ContentBlock{
		Type:   anthropic.BlockImage,
		Source: &anthropic.ImageSource{Type: anthropic.SourceURL, URL: url},
	}, nil
}

// parseDataURL decodes a "data:<media-type>;base64,<data>" URL into a base64
// image source, validating the media type, encoding and payload.
func parseDataURL(url string) (*anthropic.ImageSource, error) {
	meta, data, found := strings.Cut(strings.TrimPrefix(url, dataURLPrefix), ",")
	if !found {
		return nil, fmt.Errorf("malformed data URL: missing comma separator")
	}
	mediaType, isBase64 := parseDataURLMeta(meta)
	if !isBase64 {
		return nil, fmt.Errorf("unsupported data URL: only base64 encoding is supported")
	}
	if mediaType == "" {
		return nil, fmt.Errorf("data URL is missing a media type")
	}
	if _, err := base64.StdEncoding.DecodeString(data); err != nil {
		return nil, fmt.Errorf("data URL payload is not valid base64: %w", err)
	}
	return &anthropic.ImageSource{
		Type:      anthropic.SourceBase64,
		MediaType: mediaType,
		Data:      data,
	}, nil
}

// parseDataURLMeta extracts the media type and whether the base64 flag is set
// from the metadata segment of a data URL.
func parseDataURLMeta(meta string) (mediaType string, isBase64 bool) {
	for _, segment := range strings.Split(meta, ";") {
		switch {
		case segment == base64Token:
			isBase64 = true
		case strings.Contains(segment, "/"):
			mediaType = segment
		}
	}
	return mediaType, isBase64
}
