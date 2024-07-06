package configs

import (
	"io/ioutil"

	"gopkg.in/yaml.v2"
)

// Config represents the application configuration
type Config struct {
	ModulePath string `yaml:"modulePath"`
	// Add other configuration options as needed
}

// LoadConfig loads the configuration from a YAML file
func LoadConfig(path string) (*Config, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var cfg Config
	err = yaml.Unmarshal(data, &cfg)
	if err != nil {
		return nil, err
	}

	return &cfg, nil
}
