package validmodule

import (
	"reflect"

	"llm-agent/internal/core/servicelocator"
	"llm-agent/internal/interfaces"
)

type ValidModule struct {
	validService *ValidService
}

func NewValidModule() *ValidModule {
	return &ValidModule{
		validService: &ValidService{},
	}
}

func (m *ValidModule) RegisterServices(sl *servicelocator.ServiceLocator) {
	sl.Register(servicelocator.ServiceRegistration{
		Constructor:       m.validService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          servicelocator.Singleton,
		Name:              "validService",
		ConstructorParams: []interface{}{},
	})
}

func (m *ValidModule) UnregisterServices(sl *servicelocator.ServiceLocator) {
	sl.ReleaseService(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "validService")
}

type ValidService struct{}

func (s *ValidService) Constructor() interfaces.Service {
	return s
}

func (s *ValidService) DoSomething() {
	// Service implementation
}

var Module = NewValidModule()
