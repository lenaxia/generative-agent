package utils

import (
	"encoding/json"
	"fmt"
	"reflect"
)

// ParseJSON parses a JSON string into a map[string]interface{}
func ParseJSON(jsonStr string) (map[string]interface{}, error) {
	var data map[string]interface{}
	err := json.Unmarshal([]byte(jsonStr), &data)
	if err != nil {
		return nil, err
	}
	return data, nil
}

// GetStructMetadata retrieves the metadata from a struct
func GetStructMetadata(s interface{}) map[string]interface{} {
	metadata := make(map[string]interface{})
	v := reflect.ValueOf(s)
	t := v.Type()

	for i := 0; i < v.NumField(); i++ {
		field := t.Field(i)
		value := v.Field(i).Interface()
		metadata[field.Name] = value
	}

	return metadata
}

// PrintJSON prints a map[string]interface{} as JSON
func PrintJSON(data map[string]interface{}) {
	jsonBytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		fmt.Println("Error marshaling JSON:", err)
		return
	}
	fmt.Println(string(jsonBytes))
}
