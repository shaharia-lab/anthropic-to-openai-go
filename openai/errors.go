package openai

// Error type identifiers returned to clients.
const (
	ErrorTypeInvalidRequest = "invalid_request_error"
	ErrorTypeAuthentication = "authentication_error"
	ErrorTypeUpstream       = "upstream_error"
)

// ErrorResponse is the OpenAI error envelope.
type ErrorResponse struct {
	Error ErrorBody `json:"error"`
}

// ErrorBody describes a single error.
type ErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
	Param   string `json:"param,omitempty"`
	Code    string `json:"code,omitempty"`
}

// NewErrorResponse builds an error envelope with the given message and type.
func NewErrorResponse(message, errType string) ErrorResponse {
	return ErrorResponse{Error: ErrorBody{Message: message, Type: errType}}
}
