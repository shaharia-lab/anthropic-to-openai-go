package anthropic2openai

import (
	"crypto/rand"
	"encoding/hex"
)

// completionIDPrefix mirrors the prefix OpenAI uses for completion IDs.
const completionIDPrefix = "chatcmpl-"

// randomIDBytes is the number of random bytes in a generated identifier.
const randomIDBytes = 16

// completionID derives an OpenAI-style completion ID from an upstream message
// ID, generating a random one when the upstream ID is empty.
func completionID(upstreamID string) string {
	if upstreamID == "" {
		return completionIDPrefix + randomID()
	}
	return completionIDPrefix + upstreamID
}

// randomID returns a hex-encoded, cryptographically random identifier. A
// crypto source is used so identifiers are unpredictable; on the practically
// impossible read failure it falls back to a fixed zero string rather than
// panicking.
func randomID() string {
	buf := make([]byte, randomIDBytes)
	if _, err := rand.Read(buf); err != nil {
		return "00000000000000000000000000000000"
	}
	return hex.EncodeToString(buf)
}
