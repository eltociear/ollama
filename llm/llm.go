package llm

import (
	"fmt"
	"os"

	"github.com/jmorganca/ollama/api"
)

type LLM interface {
	Predict([]int, string, func(api.GenerateResponse)) error
	Close()
}

func New(model string, opts api.Options) (LLM, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	f, err := os.Open(model)
	if err != nil {
		return nil, err
	}

	ggml, err := DecodeGGML(f, "llama")
	if err != nil {
		return nil, err
	}

	switch ggml.ModelType {
	case "llama":
		return newLlama(model, opts)
	default:
		return nil, fmt.Errorf("unknown ggml type: %s", ggml.ModelType)
	}
}
