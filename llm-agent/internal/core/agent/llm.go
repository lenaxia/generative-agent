package agent

import (
	"llm-agent/internal/interfaces"
	"llm-agent/internal/types"
)

// generatePlan generates a plan using the LLM backend
func (a *GenerativeAgent) generatePlan(input interface{}, services []interfaces.Service) *types.Plan {
	// Get the LLM backend service
	llmService, err := a.serviceLocator.Get(reflect.TypeOf((*LLMBackend)(nil)).Elem(), "llm")
	if err != nil {
		// Handle error
		return nil
	}

	llmBackend, ok := llmService.(LLMBackend)
	if !ok {
		// Handle error
		return nil
	}

	// Call the LLM backend to generate a plan
	plan, err := llmBackend.GeneratePlan(input, services)
	if err != nil {
		// Handle error
		return nil
	}

	return plan
}

// executePlan executes a plan using the LLM backend
func (a *GenerativeAgent) executePlan(plan *types.Plan) *types.Result {
	// Get the LLM backend service
	llmService, err := a.serviceLocator.Get(reflect.TypeOf((*LLMBackend)(nil)).Elem(), "llm")
	if err != nil {
		// Handle error
		return nil
	}

	llmBackend, ok := llmService.(LLMBackend)
	if !ok {
		// Handle error
		return nil
	}

	// Execute the plan by calling the LLM backend for each step
	var result *types.Result
	for _, step := range plan.Steps {
		result, err = llmBackend.ExecuteFunction(step.Function, step.Input)
		if err != nil {
			// Handle error
			break
		}
	}

	return result
}
