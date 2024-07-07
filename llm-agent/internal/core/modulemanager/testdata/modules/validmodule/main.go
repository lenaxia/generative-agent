package main

import (
	"llm-agent/internal/core/servicelocator"
	"llm-agent/internal/interfaces"
	"llm-agent/internal/interfaces/mocks"
)

type validModule struct{}

func (m *validModule) RegisterServices(sl *servicelocator.ServiceLocator) {
	sl.Register(servicelocator.ServiceRegistration{
		Service:    &mocks.MockService{},
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:   servicelocator.Singleton,
		Name:       "validService",
	})
}

var Module interfaces.Module = &validModule{}
