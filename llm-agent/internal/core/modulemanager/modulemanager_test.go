package modulemanager

import (
	"testing"

	"llm-agent/internal/core/servicelocator"
	"llm-agent/internal/interfaces/mocks"
)

func TestModuleManager_LoadModule(t *testing.T) {
	// Create a new ServiceLocator
	sl := servicelocator.NewServiceLocator()

	// Create a new ModuleManager
	mm, err := NewModuleManager(sl, "testdata/modules")
	if err != nil {
		t.Fatalf("Failed to create ModuleManager: %v", err)
	}

	// Load a valid module
	mm.loadModule("validmodule.so")

	// Check if the module's services are registered
	service, err := sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "validService")
	if err != nil {
		t.Errorf("Failed to get service from valid module: %v", err)
	}

	// Check if the service is of the expected type
	_, ok := service.(*mocks.MockService)
	if !ok {
		t.Errorf("Service is not of the expected type")
	}
}

func TestModuleManager_UnloadModule(t *testing.T) {
	// Create a new ServiceLocator
	sl := servicelocator.NewServiceLocator()

	// Create a new ModuleManager
	mm, err := NewModuleManager(sl, "testdata/modules")
	if err != nil {
		t.Fatalf("Failed to create ModuleManager: %v", err)
	}

	// Load a valid module
	mm.loadModule("validmodule.so")

	// Unload the module
	mm.unloadModule("validmodule.so")

	// Check if the module's services are unregistered
	_, err = sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "validService")
	if err == nil {
		t.Errorf("Service from unloaded module is still registered")
	}
}

func TestModuleManager_LoadInvalidModule(t *testing.T) {
	// Create a new ServiceLocator
	sl := servicelocator.NewServiceLocator()

	// Create a new ModuleManager
	mm, err := NewModuleManager(sl, "testdata/modules")
	if err != nil {
		t.Fatalf("Failed to create ModuleManager: %v", err)
	}

	// Load an invalid module
	mm.loadModule("invalidmodule.so")

	// Check if no services are registered
	services := sl.GetServices()
	if len(services) != 0 {
		t.Errorf("Invalid module registered services")
	}
}

func TestModuleManager_WatchModules(t *testing.T) {
	// Create a new ServiceLocator
	sl := servicelocator.NewServiceLocator()

	// Create a new ModuleManager
	mm, err := NewModuleManager(sl, "testdata/modules")
	if err != nil {
		t.Fatalf("Failed to create ModuleManager: %v", err)
	}

	// Load a valid module
	mm.loadModule("validmodule.so")

	// Check if the module's services are registered
	service, err := sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "validService")
	if err != nil {
		t.Errorf("Failed to get service from valid module: %v", err)
	}

	// Check if the service is of the expected type
	_, ok := service.(*mocks.MockService)
	if !ok {
		t.Errorf("Service is not of the expected type")
	}

	// Unload the module
	mm.unloadModule("validmodule.so")

	// Check if the module's services are unregistered
	_, err = sl.Get(reflect.TypeOf((*interfaces.Service)(nil)).Elem(), "validService")
	if err == nil {
		t.Errorf("Service from unloaded module is still registered")
	}
}
