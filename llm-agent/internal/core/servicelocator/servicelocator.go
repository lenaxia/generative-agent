package servicelocator

import (
	"fmt"
	"reflect"
	"sync"

	"llm-agent/internal/interfaces"
)

// ServiceLocator is responsible for managing service registrations and instances
type ServiceLocator struct {
	registrations map[reflect.Type][]ServiceRegistration
	instances     map[reflect.Type]map[string]interfaces.Service
	mutex         sync.RWMutex
}

// ServiceRegistration holds the information needed to register a service
type ServiceRegistration struct {
	Service    interfaces.Service
	Interfaces []reflect.Type
	Lifetime   ServiceLifetime
	Name       string
}

// ServiceLifetime represents the lifetime of a service
type ServiceLifetime int

const (
	Singleton ServiceLifetime = iota
	Static
	Transient
)

// NewServiceLocator creates a new instance of ServiceLocator
func NewServiceLocator() *ServiceLocator {
	return &ServiceLocator{
		registrations: make(map[reflect.Type][]ServiceRegistration),
		instances:     make(map[reflect.Type]map[string]interfaces.Service),
	}
}

// Register registers a service with the service locator
func (sl *ServiceLocator) Register(reg ServiceRegistration) {
	sl.mutex.Lock()
	defer sl.mutex.Unlock()

	for _, iface := range reg.Interfaces {
		sl.registrations[iface] = append(sl.registrations[iface], reg)
		if reg.Lifetime == Singleton || reg.Lifetime == Static {
			sl.instances[iface] = map[string]interfaces.Service{
				reg.Name: reg.Service,
			}
		}
	}
}

// Get retrieves a service instance from the service locator
func (sl *ServiceLocator) Get(iface reflect.Type, name string) (interfaces.Service, error) {
	sl.mutex.RLock()
	defer sl.mutex.RUnlock()

	if instances, ok := sl.instances[iface]; ok {
		if instance, ok := instances[name]; ok {
			return instance, nil
		}
	}

	for _, reg := range sl.registrations[iface] {
		if reg.Lifetime == Transient || (reg.Lifetime == Singleton && name == reg.Name) {
			sl.instances[iface] = map[string]interfaces.Service{
				reg.Name: reg.Service,
            }
			return reg.Service, nil
		}
	}

	return nil, fmt.Errorf("service not found: %s", name)
}

// GetServices returns all registered services
func (sl *ServiceLocator) GetServices() []interfaces.Service {
	sl.mutex.RLock()
	defer sl.mutex.RUnlock()

	var services []interfaces.Service
	for _, regs := range sl.registrations {
		for _, reg := range regs {
			services = append(services, reg.Service)
		}
	}

	return services
}
