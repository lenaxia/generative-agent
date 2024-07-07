package servicelocator

import (
	"reflect"
	"testing"

	"llm-agent/internal/interfaces"
	"llm-agent/internal/interfaces/mocks"
)

// TestServiceLocator_Register tests the Register method of the ServiceLocator
func TestServiceLocator_Register(t *testing.T) {
	sl := NewServiceLocator()

	// Test registering a singleton service
	singletonService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Service:    singletonService,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:   Singleton,
		Name:       "singleton",
	})

	// Test registering a static service
	staticService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Service:    staticService,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:   Static,
		Name:       "static",
	})

	// Test registering a transient service
	transientService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Service:    transientService,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:   Transient,
		Name:       "transient",
	})

	// Test registering multiple services for the same interface
	anotherService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Service:    anotherService,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:   Transient,
		Name:       "another",
	})

	// Test registering a service for multiple interfaces
	multiInterfaceService := &mocks.MockMultiInterfaceService{}
	sl.Register(ServiceRegistration{
		Service:    multiInterfaceService,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem(), reflect.TypeOf((*interfaces.AnotherInterface)(nil)).Elem()},
		Lifetime:   Singleton,
		Name:       "multi",
	})

	// Verify that the services are registered correctly
	if len(sl.registrations[reflect.TypeOf((*interfaces.Service)(nil)).Elem()]) != 5 {
		t.Errorf("Expected 5 services registered for Service interface, got %d", len(sl.registrations[reflect.TypeOf((*interfaces.Service)(nil)).Elem()]))
	}

	if len(sl.registrations[reflect.TypeOf((*interfaces.AnotherInterface)(nil)).Elem()]) != 1 {
		t.Errorf("Expected 1 service registered for AnotherInterface interface, got %d", len(sl.registrations[reflect.TypeOf((*interfaces.AnotherInterface)(nil)).Elem()]))
	}

	if len(sl.instances[reflect.TypeOf((*interfaces.Service)(nil)).Elem()]) != 3 {
		t.Errorf("Expected 3 instances for Service interface, got %d", len(sl.instances[reflect.TypeOf((*interfaces.Service)(nil)).Elem()]))
	}

	if len(sl.instances[reflect.TypeOf((*interfaces.AnotherInterface)(nil)).Elem()]) != 1 {
		t.Errorf("Expected 1 instance for AnotherInterface interface, got %d", len(sl.instances[reflect.TypeOf((*interfaces.AnotherInterface)(nil)).Elem()]))
	}
}

// TestServiceLocator_Get tests the Get method of the ServiceLocator
func TestServiceLocator_Get(t *testing.T) {
	sl := NewServiceLocator()

	// Register services
	singletonService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Service:    singletonService,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:   Singleton,
		Name:       "singleton",
	})

	staticService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Service:    staticService,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:   Static,
		Name:       "static",
	})

	transientService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Service:    transientService,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:   Transient,
		Name:       "transient",
	})

	// Test getting a singleton service
	service, err := sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "singleton")
	if err != nil {
		t.Errorf("Unexpected error getting singleton service: %v", err)
	}
	if service != singletonService {
		t.Errorf("Expected singleton service, got %v", service)
	}

	// Test getting a static service
	service, err = sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "static")
	if err != nil {
		t.Errorf("Unexpected error getting static service: %v", err)
	}
	if service != staticService {
		t.Errorf("Expected static service, got %v", service)
	}

	// Test getting a transient service
	service, err = sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "transient")
	if err != nil {
		t.Errorf("Unexpected error getting transient service: %v", err)
	}
	if service != transientService {
		t.Errorf("Expected transient service, got %v", service)
	}

	// Test getting a non-existent service
	_, err = sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "nonexistent")
	if err == nil {
		t.Errorf("Expected error getting non-existent service, got nil")
	}

	// Test getting a transient service again (should create a new instance)
	newTransientService, err := sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "transient")
	if err != nil {
		t.Errorf("Unexpected error getting transient service: %v", err)
	}
	if newTransientService == transientService {
		t.Errorf("Expected a new instance of transient service, got the same instance")
	}
}

// TestServiceLocator_GetServices tests the GetServices method of the ServiceLocator
func TestServiceLocator_GetServices(t *testing.T) {
	sl := NewServiceLocator()

	// Register services
	service1 := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Service:    service1,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:   Singleton,
		Name:       "service1",
	})

	service2 := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Service:    service2,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:   Static,
		Name:       "service2",
	})

	service3 := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Service:    service3,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:   Transient,
		Name:       "service3",
	})

	// Get all registered services
	services := sl.GetServices()

	// Verify that all services are returned
	if len(services) != 3 {
		t.Errorf("Expected 3 services, got %d", len(services))
	}

	if !containsService(services, service1) || !containsService(services, service2) || !containsService(services, service3) {
		t.Errorf("Expected services %v, %v, %v, got %v", service1, service2, service3, services)
	}
}

// TestServiceLocator_RegisterAndGet tests registering and getting services with different lifetimes
func TestServiceLocator_RegisterAndGet(t *testing.T) {
	sl := NewServiceLocator()

	// Register a singleton service
	singletonService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Service:    singletonService,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:   Singleton,
		Name:       "singleton",
	})

	// Get the singleton service
	service, err := sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "singleton")
	if err != nil {
		t.Errorf("Unexpected error getting singleton service: %v", err)
	}
	if service != singletonService {
		t.Errorf("Expected singleton service, got %v", service)
	}

	// Register a static service
	staticService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Service:    staticService,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:   Static,
		Name:       "static",
	})

	// Get the static service
	service, err = sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "static")
	if err != nil {
		t.Errorf("Unexpected error getting static service: %v", err)
	}
	if service != staticService {
		t.Errorf("Expected static service, got %v", service)
	}

	// Register a transient service
	transientService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Service:    transientService,
		Interfaces: []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:   Transient,
		Name:       "transient",
	})

	// Get the transient service
	service, err = sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "transient")
	if err != nil {
		t.Errorf("Unexpected error getting transient service: %v", err)
	}
	if service != transientService {
		t.Errorf("Expected transient service, got %v", service)
	}

	// Get the transient service again (should create a new instance)
	newTransientService, err := sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "transient")
	if err != nil {
		t.Errorf("Unexpected error getting transient service: %v", err)
	}
	if newTransientService == transientService {
		t.Errorf("Expected a new instance of transient service, got the same instance")
	}
}

// containsService is a helper function to check if a service is present in a slice of services
func containsService(services []interfaces.Service, service interfaces.Service) bool {
	for _, s := range services {
		if s == service {
			return true
		}
	}
	return false
}
