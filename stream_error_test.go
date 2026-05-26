package anthropic2openai

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/shaharia-lab/anthropic-to-openai-go/anthropic"
)

// failWriter fails on the failAt-th write (1-based), succeeding before that.
type failWriter struct {
	failAt int
	n      int
}

func (f *failWriter) Write(p []byte) (int, error) {
	f.n++
	if f.n >= f.failAt {
		return 0, errors.New("write failed")
	}
	return len(p), nil
}

// doneFailWriter records output but fails when writing the stream terminator.
type doneFailWriter struct {
	strings.Builder
}

func (w *doneFailWriter) Write(p []byte) (int, error) {
	if bytes.Contains(p, []byte(streamDone)) {
		return 0, errors.New("done write failed")
	}
	return w.Builder.Write(p)
}

func TestTranslateStreamWriteError(t *testing.T) {
	err := TranslateStream(&failWriter{failAt: 1}, strings.NewReader(textStreamFixture), StreamParams{})
	if err == nil {
		t.Fatal("expected write error to propagate")
	}
}

func TestTranslateStreamDoneWriteError(t *testing.T) {
	err := TranslateStream(&doneFailWriter{}, strings.NewReader(textStreamFixture), StreamParams{})
	if err == nil {
		t.Fatal("expected terminator write error to propagate")
	}
}

func TestTranslateStreamIgnoresEmptyAndUnknownDeltas(t *testing.T) {
	fixture := `data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}

data: {"type":"content_block_delta","index":7,"delta":{"type":"input_json_delta","partial_json":"{}"}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"unknown_delta"}}

data: {"type":"ping"}

data: {"type":"message_stop"}

`
	var out strings.Builder
	if err := TranslateStream(&out, strings.NewReader(fixture), StreamParams{}); err != nil {
		t.Fatalf("translate stream: %v", err)
	}
	chunks := collectChunks(t, out.String())
	// Only the role chunk (from emitFinish's ensureRole) and the finish chunk
	// should be present; no content deltas are emitted.
	for _, c := range chunks {
		if len(c.Choices) > 0 && c.Choices[0].Delta.Content != "" {
			t.Fatalf("unexpected content delta: %q", c.Choices[0].Delta.Content)
		}
	}
}

func TestStreamEventErrorWithoutBody(t *testing.T) {
	err := streamEventError(anthropic.StreamEvent{Type: anthropic.EventError})
	if err == nil || !strings.Contains(err.Error(), "stream error") {
		t.Fatalf("err = %v", err)
	}
}
