package embeddings

// Embedder is an interface for generating embeddings
type Embedder interface {
    // Embed generates an embedding vector for the given input text
    Embed(text string) ([]float32, error)
}

// NewEmbedder creates a new instance of an Embedder
func NewEmbedder(config interface{}) (Embedder, error) {
    // Implement the logic to create a new Embedder instance based on the provided config
    return nil, nil
}
