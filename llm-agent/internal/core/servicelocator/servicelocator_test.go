package servicelocator

import (
	"reflect"
	"sync"
	"testing"

	"llm-agent/internal/interfaces"
	"llm-agent/internal/interfaces/mocks"
)

// TestServiceLocator_Register tests the Register method of the ServiceLocator
func TestServiceLocator_Register(t *testing.T) {
	sl := NewServiceLocator()

	// Test registering a singleton service
	singletonService := &mocks.MockService{}
	err := sl.Register(ServiceRegistration{
		Constructor:       singletonService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Singleton,
		Name:              "singleton",
		ConstructorParams: []interface{}{},
	})
	if err != nil {
		t.Errorf("Unexpected error registering singleton service: %v", err)
	}

	// Test registering a static service
	staticService := &mocks.MockService{}
	err = sl.Register(ServiceRegistration{
		Constructor:       staticService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Static,
		Name:              "static",
		ConstructorParams: []interface{}{},
	})
	if err != nil {
		t.Errorf("Unexpected error registering static service: %v", err)
	}

	// Test registering a transient service
	transientService := &mocks.MockService{}
	err = sl.Register(ServiceRegistration{
		Constructor:       transientService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Transient,
		Name:              "transient",
		ConstructorParams: []interface{}{},
	})
	if err != nil {
		t.Errorf("Unexpected error registering transient service: %v", err)
	}

	// Test registering multiple services for the same interface
	anotherService := &mocks.MockService{}
	err = sl.Register(ServiceRegistration{
		Constructor:       anotherService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Transient,
		Name:              "another",
		ConstructorParams: []interface{}{},
	})
	if err != nil {
		t.Errorf("Unexpected error registering another service: %v", err)
	}

	// Test registering a service for multiple interfaces
	multiInterfaceService := &mocks.MockMultiInterfaceService{}
	err = sl.Register(ServiceRegistration{
		Constructor:       multiInterfaceService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem(), reflect.TypeOf((*interfaces.AnotherInterface)(nil)).Elem()},
		Lifetime:          Singleton,
		Name:              "multi",
		ConstructorParams: []interface{}{},
	})
	if err != nil {
		t.Errorf("Unexpected error registering multi-interface service: %v", err)
	}

	// Verify that the services are registered correctly
	if len(sl.registrations[reflect.TypeOf((*interfaces.Service)(nil)).Elem()]) != 5 {
		t.Errorf("Expected 5 services registered for Service interface, got %d", len(sl.registrations[reflect.TypeOf((*interfaces.Service)(nil)).Elem()]))
	}

	if len(sl.registrations[reflect.TypeOf((*interfaces.AnotherInterface)(nil)).Elem()]) != 1 {
		t.Errorf("Expected 1 service registered for AnotherInterface interface, got %d", len(sl.registrations[reflect.TypeOf((*interfaces.AnotherInterface)(nil)).Elem()]))
	}

	// Instances should not be created during registration
	if len(sl.instances[reflect.TypeOf((*interfaces.Service)(nil)).Elem()]) != 0 {
		t.Errorf("Expected 0 instances for Service interface, got %d", len(sl.instances[reflect.TypeOf((*interfaces.Service)(nil)).Elem()]))
	}

	if len(sl.instances[reflect.TypeOf((*interfaces.AnotherInterface)(nil)).Elem()]) != 0 {
		t.Errorf("Expected 0 instances for AnotherInterface interface, got %d", len(sl.instances[reflect.TypeOf((*interfaces.AnotherInterface)(nil)).Elem()]))
	}
}

// TestServiceLocator_Get tests the Get method of the ServiceLocator
func TestServiceLocator_Get(t *testing.T) {
	sl := NewServiceLocator()

	// Register services
	singletonService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       singletonService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Singleton,
		Name:              "singleton",
		ConstructorParams: []interface{}{},
	})

	staticService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       staticService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Static,
		Name:              "static",
		ConstructorParams: []interface{}{},
	})

	transientService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       transientService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Transient,
		Name:              "transient",
		ConstructorParams: []interface{}{},
	})

	// Test getting a singleton service
	service, err := sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "singleton")
	if err != nil {
		t.Errorf("Unexpected error getting singleton service: %v", err)
	}
	if _, ok := service.(*mocks.MockService); !ok {
		t.Errorf("Expected singleton service, got %v", service)
	}

	// Test getting a static service
	service, err = sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "static")
	if err != nil {
		t.Errorf("Unexpected error getting static service: %v", err)
	}
	if _, ok := service.(*mocks.MockService); !ok {
		t.Errorf("Expected static service, got %v", service)
	}

	// Test getting a transient service
	service, err = sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "transient")
	if err != nil {
		t.Errorf("Unexpected error getting transient service: %v", err)
	}
	if _, ok := service.(*mocks.MockService); !ok {
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
	if newTransientService == service {
		t.Errorf("Expected a new instance of transient service, got the same instance")
	}
}

// TestServiceLocator_GetServices tests the GetServices method of the ServiceLocator
func TestServiceLocator_GetServices(t *testing.T) {
	sl := NewServiceLocator()

	// Register services
	service1 := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       service1.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Singleton,
		Name:              "service1",
		ConstructorParams: []interface{}{},
	})

	service2 := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       service2.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Static,
		Name:              "service2",
		ConstructorParams: []interface{}{},
	})

	service3 := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       service3.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Transient,
		Name:              "service3",
		ConstructorParams: []interface{}{},
	})

	// Get all registered services
	services := sl.GetServices()

	// Verify that all services are returned
	if len(services) != 3 {
		t.Errorf("Expected 3 services, got %d", len(services))
	}

	foundService1, foundService2, foundService3 := false, false, false
	for _, service := range services {
		if _, ok := service.Instance.(*mocks.MockService); ok {
			if service.Instance == service1 {
				foundService1 = true
			} else if service.Instance == service2 {
				foundService2 = true
			} else if service.Instance == service3 {
				foundService3 = true
			}
		}
	}

	if !foundService1 || !foundService2 || !foundService3 {
		t.Errorf("Expected services %v, %v, %v, got %v", service1, service2, service3, services)
	}
}

// TestServiceLocator_RegisterAndGet tests registering and getting services with different lifetimes
func TestServiceLocator_RegisterAndGet(t *testing.T) {
	sl := NewServiceLocator()

	// Register a singleton service
	singletonService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       singletonService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Singleton,
		Name:              "singleton",
		ConstructorParams: []interface{}{},
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
		Constructor:       staticService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Static,
		Name:              "static",
		ConstructorParams: []interface{}{},
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
		Constructor:       transientService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Transient,
		Name:              "transient",
		ConstructorParams: []interface{}{},
	})

	// Get the transient service
	service, err = sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "transient")
	if err != nil {
		t.Errorf("Unexpected error getting transient service: %v", err)
	}
	if _, ok := service.(*mocks.MockService); !ok {
		t.Errorf("Expected transient service, got %v", service)
	}

	// Get the transient service again (should create a new instance)
	newTransientService, err := sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "transient")
	if err != nil {
		t.Errorf("Unexpected error getting transient service: %v", err)
	}
	if newTransientService == service {
		t.Errorf("Expected a new instance of transient service, got the same instance")
	}
}

// TestServiceLocator_RegisterWithEmptyName tests registering a service with an empty name
func TestServiceLocator_RegisterWithEmptyName(t *testing.T) {
	sl := NewServiceLocator()

	// Register a service with an empty name
	service := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       service.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Singleton,
		Name:              "",
		ConstructorParams: []interface{}{},
	})

	// Get the service with an empty name
	serviceInstance, err := sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "")
	if err != nil {
		t.Errorf("Unexpected error getting service with empty name: %v", err)
	}
	if _, ok := serviceInstance.(*mocks.MockService); !ok {
		t.Errorf("Expected service with empty name, got %v", serviceInstance)
	}
}

// TestServiceLocator_RegisterWithSameNameDifferentLifetimes tests registering services with the same name but different lifetimes
func TestServiceLocator_RegisterWithSameNameDifferentLifetimes(t *testing.T) {
	sl := NewServiceLocator()

	// Register a singleton service
	singletonService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       singletonService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Singleton,
		Name:              "myService",
		ConstructorParams: []interface{}{},
	})

	// Register a static service with the same name
	staticService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       staticService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Static,
		Name:              "myService",
		ConstructorParams: []interface{}{},
	})

	// Register a transient service with the same name
	transientService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       transientService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Transient,
		Name:              "myService",
		ConstructorParams: []interface{}{},
	})

	// Get the singleton service
	service, err := sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "myService")
	if err != nil {
		t.Errorf("Unexpected error getting singleton service: %v", err)
	}
	if service != singletonService {
		t.Errorf("Expected singleton service, got %v", service)
	}

	// Get the static service
	service, err = sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "myService")
	if err != nil {
		t.Errorf("Unexpected error getting static service: %v", err)
	}
	if service != staticService {
		t.Errorf("Expected static service, got %v", service)
	}

	// Get the transient service
	service, err = sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "myService")
	if err != nil {
		t.Errorf("Unexpected error getting transient service: %v", err)
	}
	if _, ok := service.(*mocks.MockService); !ok {
		t.Errorf("Expected transient service, got %v", service)
	}
}

// TestServiceLocator_RegisterWithSameNameAndLifetime tests registering services with the same name and lifetime
func TestServiceLocator_RegisterWithSameNameAndLifetime(t *testing.T) {
	sl := NewServiceLocator()

	// Register a singleton service
	singletonService := &mocks.MockService{}
	err := sl.Register(ServiceRegistration{
		Constructor:       singletonService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Singleton,
		Name:              "myService",
		ConstructorParams: []interface{}{},
	})
	if err != nil {
		t.Errorf("Unexpected error registering singleton service: %v", err)
	}

	// Register another singleton service with the same name
	anotherSingletonService := &mocks.MockService{}
	err = sl.Register(ServiceRegistration{
		Constructor:       anotherSingletonService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Singleton,
		Name:              "myService",
		ConstructorParams: []interface{}{},
	})

	if err == nil {
		t.Errorf("Expected error when registering service with same name and lifetime, got nil")
	}
}

// TestServiceLocator_GetWithIncorrectInterfaceType tests getting a service with an incorrect interface type
func TestServiceLocator_GetWithIncorrectInterfaceType(t *testing.T) {
	sl := NewServiceLocator()

	// Register a service
	service := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       service.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Singleton,
		Name:              "myService",
		ConstructorParams: []interface{}{},
	})

	// Try to get the service with an incorrect interface type
	_, err := sl.Get(reflect.TypeOf((*interfaces.AnotherInterface)(nil)).Elem(), "myService")
	if err == nil {
		t.Errorf("Expected error when getting service with incorrect interface type, got nil")
	}
}

// TestServiceLocator_Concurrency tests concurrent access to the service locator
func TestServiceLocator_Concurrency(t *testing.T) {
	sl := NewServiceLocator()

	// Register a singleton service
	singletonService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       singletonService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Singleton,
		Name:              "singleton",
		ConstructorParams: []interface{}{},
	})

	// Register a static service
	staticService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       staticService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Static,
		Name:              "static",
		ConstructorParams: []interface{}{},
	})

	// Register a transient service
	transientService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       transientService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Transient,
		Name:              "transient",
		ConstructorParams: []interface{}{},
	})

	var wg sync.WaitGroup
	wg.Add(100)

	for i := 0; i < 100; i++ {
		go func() {
			defer wg.Done()

			// Get the singleton service
			service, err := sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "singleton")
			if err != nil {
				t.Errorf("Unexpected error getting singleton service: %v", err)
			}
			if service != singletonService {
				t.Errorf("Expected singleton service, got %v", service)
			}

			// Get the static service
			service, err = sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "static")
			if err != nil {
				t.Errorf("Unexpected error getting static service: %v", err)
			}
			if service != staticService {
				t.Errorf("Expected static service, got %v", service)
			}

			// Get the transient service
			service, err = sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "transient")
			if err != nil {
				t.Errorf("Unexpected error getting transient service: %v", err)
			}
			if _, ok := service.(*mocks.MockService); !ok {
				t.Errorf("Expected transient service, got %v", service)
			}
		}()
	}

	wg.Wait()
}

// TestServiceLocator_ReleaseService tests the ReleaseService method
func TestServiceLocator_ReleaseService(t *testing.T) {
	sl := NewServiceLocator()

	// Register a disposable singleton service
	disposableService := &mocks.MockDisposableService{}
	sl.Register(ServiceRegistration{
		Constructor:       disposableService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem(), reflect.TypeOf((*interfaces.Disposable)(nil)).Elem()},
		Lifetime:          Singleton,
		Name:              "disposable",
		ConstructorParams: []interface{}{},
	})

	// Get the disposable service
	service, err := sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "disposable")
	if err != nil {
		t.Errorf("Unexpected error getting disposable service: %v", err)
	}

	// Release the service
	sl.ReleaseService(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "disposable")

	// Check if the Dispose method was called
	if !disposableService.DisposeCalled {
		t.Errorf("Expected Dispose method to be called, but it wasn't")
	}
}

// TestServiceLocator_NestedServiceLocator tests the NestedServiceLocator
func TestServiceLocator_NestedServiceLocator(t *testing.T) {
	sl := NewServiceLocator()

	// Register a service in the root service locator
	rootService := &mocks.MockService{}
	sl.Register(ServiceRegistration{
		Constructor:       rootService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Singleton,
		Name:              "root",
		ConstructorParams: []interface{}{},
	})

	// Create a nested service locator
	nestedSL := NewNestedServiceLocator(sl)

	// Register a service in the nested service locator
	nestedService := &mocks.MockService{}
	nestedSL.Register(ServiceRegistration{
		Constructor:       nestedService.Constructor,
		Interfaces:        []reflect.Type{reflect.TypeOf((*interfaces.Service)(nil)).Elem()},
		Lifetime:          Singleton,
		Name:              "nested",
		ConstructorParams: []interface{}{},
	})

	// Get the service from the nested service locator
	service, err := nestedSL.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "nested")
	if err != nil {
		t.Errorf("Unexpected error getting service from nested service locator: %v", err)
	}
	if service != nestedService {
		t.Errorf("Expected nested service, got %v", service)
	}

	// Get the service from the root service locator
	service, err = nestedSL.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "root")
	if err != nil {
		t.Errorf("Unexpected error getting service from root service locator: %v", err)
	}
	if service != rootService {
		t.Errorf("Expected root service, got %v", service)
	}

	// Get all services from the nested service locator
	services := nestedSL.GetServices()
	if len(services) != 2 {
		t.Errorf("Expected 2 services, got %d", len(services))
	}

	// Release a service from the nested service locator
	nestedSL.ReleaseService(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "nested")

	// Release a service from the root service locator
	nestedSL.ReleaseService(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "root")
}

// containsService is a helper function to check if a service is present in a slice of services
func containsService(services []Service, service interface{}) bool {
	for _, s := range services {
		if s.Instance == service {
			return true
		}
	}
	return false
}
