package invalidmodule

import (
	"fmt"

	"llm-agent/internal/core/servicelocator"
)

type InvalidModule struct{}

func (m *InvalidModule) RegisterServices(sl *servicelocator.ServiceLocator) {
	fmt.Println("Invalid module")
}

var Module = &InvalidModule{}
