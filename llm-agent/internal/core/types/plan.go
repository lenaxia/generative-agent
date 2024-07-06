package types

// Plan represents a plan for the generative agent to execute
type Plan struct {
	Steps []*PlanStep
}

// PlanStep represents a step in a plan
type PlanStep struct {
	Function Function
	Input    interface{}
}
