package main

import (
	"fmt"
)

type invalidModule struct{}

func (m *invalidModule) RegisterServices() {
	fmt.Println("Invalid module")
}

var Module = &invalidModule{}
