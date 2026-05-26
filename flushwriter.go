package anthropic2openai

import (
	"io"
	"net/http"
)

// flushWriter adapts an http.ResponseWriter into an io.Writer that flushes after
// each write when the underlying writer supports http.Flusher. This lets the
// stream translator push SSE chunks to clients without buffering.
type flushWriter struct {
	w       io.Writer
	flusher http.Flusher
}

// newFlushWriter wraps w, capturing its Flusher when available.
func newFlushWriter(w http.ResponseWriter) *flushWriter {
	fw := &flushWriter{w: w}
	if f, ok := w.(http.Flusher); ok {
		fw.flusher = f
	}
	return fw
}

// Write forwards to the underlying writer.
func (f *flushWriter) Write(p []byte) (int, error) {
	return f.w.Write(p)
}

// Flush flushes the underlying writer when it supports flushing.
func (f *flushWriter) Flush() {
	if f.flusher != nil {
		f.flusher.Flush()
	}
}
