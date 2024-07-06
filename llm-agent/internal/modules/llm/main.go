package main

import (
	"llm-agent/internal/core/servicelocator"
	"llm-agent/internal/interfaces"
	"llm-agent/internal/modules/llm"
	"reflect"
)

// Module is the module instance
var Module interfaces.Module = &llm.LLMModule{}

// RegisterServices registers the services provided by this module
func (m *llm.LLMModule) RegisterServices(sl interfaces.ServiceRegistry) {
	sl.Register(servicelocator.ServiceRegistration{
		Service:    m,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem(), reflect.TypeOf((*interfaces.LLMBackend)(nil)).Elem()},
		Lifetime:   servicelocator.Singleton,
		Name:       "llm",
	})
}
