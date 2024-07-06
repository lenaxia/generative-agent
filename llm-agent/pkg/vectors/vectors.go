package vectors

// VectorSimilarity is an interface for calculating vector similarity
type VectorSimilarity interface {
    // Similarity calculates the similarity between two vectors
    Similarity(vec1, vec2 []float32) float32
}

// NewVectorSimilarity creates a new instance of a VectorSimilarity
func NewVectorSimilarity(config interface{}) (VectorSimilarity, error) {
    // Implement the logic to create a new VectorSimilarity instance based on the provided config
    return nil, nil
}
