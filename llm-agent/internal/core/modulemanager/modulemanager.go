package modulemanager

import (
	"fmt"
	"path/filepath"
	"plugin"
	"sync"

	"llm-agent/internal/core/servicelocator"
	"llm-agent/internal/interfaces"

	"github.com/fsnotify/fsnotify"
)

// ModuleManager is responsible for managing modules
type ModuleManager struct {
	serviceLocator *servicelocator.ServiceLocator
	modulePath     string
	modules        []interfaces.Module
	watcher        *fsnotify.Watcher
	mutex          sync.Mutex
}

// NewModuleManager creates a new instance of ModuleManager
func NewModuleManager(serviceLocator *servicelocator.ServiceLocator, modulePath string) (*ModuleManager, error) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}

	err = watcher.Add(modulePath)
	if err != nil {
		return nil, err
	}

	mm := &ModuleManager{
		serviceLocator: serviceLocator,
		modulePath:     modulePath,
		watcher:        watcher,
	}

	go mm.watchModules()

	return mm, nil
}

// watchModules watches the module path for changes and loads/unloads modules accordingly
func (mm *ModuleManager) watchModules() {
	for {
		select {
		case event, ok := <-mm.watcher.Events:
			if !ok {
				return
			}

			if event.Op&fsnotify.Create == fsnotify.Create {
				mm.loadModule(filepath.Base(event.Name))
			} else if event.Op&fsnotify.Remove == fsnotify.Remove {
				mm.unloadModule(filepath.Base(event.Name))
			}
		case err, ok := <-mm.watcher.Errors:
			if !ok {
				return
			}
			fmt.Println("Error watching modules:", err)
		}
	}
}

// loadModule loads a module and registers its services
func (mm *ModuleManager) loadModule(moduleName string) {
	mm.mutex.Lock()
	defer mm.mutex.Unlock()

	modulePath := filepath.Join(mm.modulePath, moduleName)
	module, err := loadModule(modulePath)
	if err != nil {
		fmt.Println("Error loading module:", err)
		return
	}

	module.RegisterServices(mm.serviceLocator)
	mm.modules = append(mm.modules, module)
	fmt.Printf("Loaded module: %s\n", moduleName)
}

// unloadModule unloads a module and removes its services
func (mm *ModuleManager) unloadModule(moduleName string) {
	mm.mutex.Lock()
	defer mm.mutex.Unlock()

	for i, module := range mm.modules {
		if filepath.Base(reflect.TypeOf(module).PkgPath()) == moduleName {
			mm.modules = append(mm.modules[:i], mm.modules[i+1:]...)
			fmt.Printf("Unloaded module: %s\n", moduleName)
			return
		}
	}
}

// loadModule loads a module from a file
func loadModule(modulePath string) (interfaces.Module, error) {
	module, err := plugin.Open(modulePath)
	if err != nil {
		return nil, err
	}

	symbol, err := module.Lookup("Module")
	if err != nil {
		return nil, err
	}

	mod, ok := symbol.(interfaces.Module)
	if !ok {
		return nil, fmt.Errorf("module %s does not implement Module interface", modulePath)
	}

	return mod, nil
}
