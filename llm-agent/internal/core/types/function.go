package types

// Function represents a function that can be called by the generative agent
type Function struct {
	Name        string
	Description string
	Parameters  []Parameter
}

// Parameter represents a parameter of a function
type Parameter struct {
	Name        string
	Description string
	Type        string
}
