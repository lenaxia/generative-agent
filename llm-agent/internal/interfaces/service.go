package interfaces

import "llm-agent/internal/types"

// Service is the base interface for all services
type Service interface {
	GetMetadata() types.ServiceMetadata
}

// LLMBackend is an interface for interacting with a Large Language Model
type LLMBackend interface {
	Service
	GeneratePlan(input interface{}, services []Service) (*types.Plan, error)
	ExecuteFunction(function *types.Function, input interface{}) (*types.Result, error)
}
