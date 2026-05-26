package anthropic2openai

// intPtr returns a pointer to v, for building requests with optional fields.
func intPtr(v int) *int { return &v }

// floatPtr returns a pointer to v, for building requests with optional fields.
func floatPtr(v float64) *float64 { return &v }
