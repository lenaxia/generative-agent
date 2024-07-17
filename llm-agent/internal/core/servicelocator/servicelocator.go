package servicelocator

import (
	"fmt"
	"reflect"
	"sync"
)

// ServiceLifetime represents the lifetime of a service
type ServiceLifetime int

const (
	Singleton ServiceLifetime = iota
	Static
	Transient
)

// ServiceRegistration holds the information needed to register a service
type ServiceRegistration struct {
	Constructor       interface{}
	Interfaces        []reflect.Type
	Lifetime          ServiceLifetime
	Name              string
	ConstructorParams []interface{}
}

// Service represents a registered service instance
type Service struct {
	Instance interface{}
	Dispose  func()
}

// ServiceLocator is responsible for managing service registrations and instances
type ServiceLocator struct {
	registrations map[reflect.Type][]ServiceRegistration
	instances     map[reflect.Type]map[string]*Service
	mutex         sync.RWMutex
	lifetimes     map[ServiceLifetime]LifetimeManager
}

// LifetimeManager manages the creation and disposal of service instances based on their lifetime
type LifetimeManager interface {
	GetInstance(reg ServiceRegistration) (*Service, error)
	ReleaseInstance(service *Service)
}

// NewServiceLocator creates a new instance of ServiceLocator
func NewServiceLocator() *ServiceLocator {
	sl := &ServiceLocator{
		registrations: make(map[reflect.Type][]ServiceRegistration),
		instances:     make(map[reflect.Type]map[string]*Service),
		lifetimes: map[ServiceLifetime]LifetimeManager{
			Singleton: &singletonLifetimeManager{},
			Static:    &staticLifetimeManager{},
			Transient: &transientLifetimeManager{},
		},
	}
	return sl
}

// Register registers a service with the service locator
func (sl *ServiceLocator) Register(reg ServiceRegistration) error {
	sl.mutex.Lock()
	defer sl.mutex.Unlock()

	for _, iface := range reg.Interfaces {
		for _, existingReg := range sl.registrations[iface] {
			if existingReg.Constructor == reg.Constructor {
				return fmt.Errorf("service already registered: %v", reg.Constructor)
			}
		}

		sl.registrations[iface] = append(sl.registrations[iface], reg)
	}

	return nil
}

// Get retrieves a service instance from the service locator
func (sl *ServiceLocator) Get(iface reflect.Type, name string) (interface{}, error) {
	sl.mutex.RLock()
	defer sl.mutex.RUnlock()

	if instances, ok := sl.instances[iface]; ok {
		if instance, ok := instances[name]; ok {
			return instance.Instance, nil
		}
	}

	for _, reg := range sl.registrations[iface] {
		if reg.Name == name || name == "" {
			instance, err := sl.lifetimes[reg.Lifetime].GetInstance(reg)
			if err != nil {
				return nil, err
			}
			sl.instances[iface] = map[string]*Service{
				reg.Name: instance,
			}
			return instance.Instance, nil
		}
	}

	return nil, fmt.Errorf("service not found: %s", name)
}

// GetServices returns all registered services
func (sl *ServiceLocator) GetServices() []Service {
	sl.mutex.RLock()
	defer sl.mutex.RUnlock()

	var services []Service
	for _, regs := range sl.registrations {
		for _, reg := range regs {
			instance, err := sl.lifetimes[reg.Lifetime].GetInstance(reg)
			if err != nil {
				// Handle error or log
				continue
			}
			services = append(services, *instance)
		}
	}

	return services
}

// ReleaseService releases a service instance
func (sl *ServiceLocator) ReleaseService(iface reflect.Type, name string) {
	sl.mutex.Lock()
	defer sl.mutex.Unlock()

	if instances, ok := sl.instances[iface]; ok {
		if instance, ok := instances[name]; ok {
			sl.lifetimes[instance.Lifetime].ReleaseInstance(instance)
			delete(instances, name)
		}
	}
}

// singletonLifetimeManager manages singleton service instances
type singletonLifetimeManager struct {
	instances map[reflect.Type]*Service
	mutex     sync.Mutex
}

func (m *singletonLifetimeManager) GetInstance(reg ServiceRegistration) (*Service, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if instance, ok := m.instances[reg.Constructor]; ok {
		return instance, nil
	}

	instance, err := createInstance(reg)
	if err != nil {
		return nil, err
	}

	m.instances[reg.Constructor] = instance
	return instance, nil
}

func (m *singletonLifetimeManager) ReleaseInstance(service *Service) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if service.Dispose != nil {
		service.Dispose()
	}
}

// staticLifetimeManager manages static service instances
type staticLifetimeManager struct {
	instances map[reflect.Type]*Service
	mutex     sync.Mutex
}

func (m *staticLifetimeManager) GetInstance(reg ServiceRegistration) (*Service, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if instance, ok := m.instances[reg.Constructor]; ok {
		return instance, nil
	}

	instance, err := createInstance(reg)
	if err != nil {
		return nil, err
	}

	m.instances[reg.Constructor] = instance
	return instance, nil
}

func (m *staticLifetimeManager) ReleaseInstance(service *Service) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if service.Dispose != nil {
		service.Dispose()
	}
}

// transientLifetimeManager manages transient service instances
type transientLifetimeManager struct{}

func (m *transientLifetimeManager) GetInstance(reg ServiceRegistration) (*Service, error) {
	instance, err := createInstance(reg)
	if err != nil {
		return nil, err
	}

	return instance, nil
}

func (m *transientLifetimeManager) ReleaseInstance(service *Service) {
	if service.Dispose != nil {
		service.Dispose()
	}
}

// createInstance creates a new service instance using the provided constructor and parameters
func createInstance(reg ServiceRegistration) (*Service, error) {
	constructorValue := reflect.ValueOf(reg.Constructor)
	params := make([]reflect.Value, len(reg.ConstructorParams))
	for i, param := range reg.ConstructorParams {
		params[i] = reflect.ValueOf(param)
	}

	instance := constructorValue.Call(params)
	if len(instance) == 0 {
		return nil, fmt.Errorf("failed to create instance of %v", reg.Constructor)
	}

	var dispose func()
	if disposer, ok := instance[0].Interface().(Disposable); ok {
		dispose = disposer.Dispose
	}

	return &Service{
		Instance: instance[0].Interface(),
		Dispose:  dispose,
	}, nil
}

// Disposable is an interface for services that need to be disposed
type Disposable interface {
	Dispose()
}

// NestedServiceLocator is a service locator that can be nested within another service locator
type NestedServiceLocator struct {
	*ServiceLocator
	parent *ServiceLocator
}

// NewNestedServiceLocator creates a new nested service locator
func NewNestedServiceLocator(parent *ServiceLocator) *NestedServiceLocator {
	return &NestedServiceLocator{
		ServiceLocator: NewServiceLocator(),
		parent:         parent,
	}
}

// Get retrieves a service instance from the nested service locator
func (nsl *NestedServiceLocator) Get(iface reflect.Type, name string) (interface{}, error) {
	// First, try to get the service from the nested service locator
	instance, err := nsl.ServiceLocator.Get(iface, name)
	if err == nil {
		return instance, nil
	}

	// If not found, try to get the service from the parent service locator
	return nsl.parent.Get(iface, name)
}

// GetServices returns all registered services in the nested service locator and its parent
func (nsl *NestedServiceLocator) GetServices() []Service {
	services := nsl.ServiceLocator.GetServices()
	parentServices := nsl.parent.GetServices()
	return append(services, parentServices...)
}

// ReleaseService releases a service instance from the nested service locator or its parent
func (nsl *NestedServiceLocator) ReleaseService(iface reflect.Type, name string) {
	if instances, ok := nsl.instances[iface]; ok {
		if instance, ok := instances[name]; ok {
			nsl.lifetimes[instance.Lifetime].ReleaseInstance(instance)
			delete(instances, name)
			return
		}
	}

	nsl.parent.ReleaseService(iface, name)
}
