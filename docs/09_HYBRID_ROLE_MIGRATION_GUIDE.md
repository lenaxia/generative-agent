# Hybrid Role Migration Guide

This guide explains how to migrate existing LLM roles to the new hybrid role lifecycle architecture and create new hybrid roles from scratch.

## Overview

The hybrid role lifecycle architecture unifies LLM and programmatic execution into a single, configurable pattern with:

- **Pre-processing hooks**: Fetch data before LLM execution
- **Post-processing hooks**: Format and enhance LLM results
- **Parameter extraction**: Automatic parameter parsing during routing
- **Data injection**: Pre-processed data injected into LLM context

## Migration Process

### Step 1: Update Role Definition

Convert your existing role definition to include hybrid configuration:

```yaml
# Before (LLM-only role)
role:
  name: "weather"
  version: "1.0.0"
  description: "Weather information specialist"
  fast_reply: true

tools:
  shared:
    - "get_weather"
    - "get_weather_forecast"

prompts:
  system: |
    You are a weather specialist. Use the weather tools to get current data.
```

```yaml
# After (Hybrid role)
role:
  name: "weather"
  version: "2.0.0"
  description: "Weather role with pre-processing data fetching"
  execution_type: "hybrid" # NEW: Specify hybrid execution
  fast_reply: true

# NEW: Parameter schema for routing extraction
parameters:
  location:
    type: "string"
    required: true
    description: "City, state, country, or coordinates"
    examples: ["Seattle", "New York, NY", "90210"]

  timeframe:
    type: "string"
    required: false
    enum: ["current", "today", "tomorrow", "this week"]
    default: "current"

# NEW: Lifecycle hooks configuration
lifecycle:
  pre_processing:
    enabled: true
    functions:
      - name: "fetch_weather_data"
        uses_parameters: ["location", "timeframe"]

  post_processing:
    enabled: true
    functions:
      - name: "format_for_tts"

# UPDATED: System prompt with pre-processed data
prompts:
  system: |
    You are a weather specialist. Weather data has been pre-fetched:
    - Current weather: {weather_current}
    - Location: {location_resolved}

    Interpret this data naturally for the user.

# UPDATED: Remove tools since data is pre-fetched
tools:
  automatic: false
  shared: [] # No tools needed - data is pre-fetched
```

### Step 2: Create Lifecycle Functions

Create `roles/{role_name}/lifecycle.py` with your pre and post-processing functions:

```python
# roles/weather/lifecycle.py

import asyncio
from datetime import datetime
from typing import Dict, Any
from roles.shared_tools.weather_tools import get_weather
from common.task_context import TaskContext
import logging

logger = logging.getLogger(__name__)


async def fetch_weather_data(instruction: str, context: TaskContext,
                           parameters: Dict) -> Dict[str, Any]:
    """
    Pre-processor: Fetch weather data before LLM call.

    Args:
        instruction: Original user instruction
        context: Task context
        parameters: Extracted parameters from routing

    Returns:
        Dict containing weather data for LLM context
    """
    location = parameters.get("location")
    timeframe = parameters.get("timeframe", "current")

    if not location:
        raise ValueError("Location parameter is required")

    # Fetch weather data using existing tools
    weather_result = get_weather(location)

    if weather_result.get("status") == "error":
        raise ValueError(f"Weather API error: {weather_result.get('error')}")

    return {
        "weather_current": weather_result.get("weather", {}),
        "location_resolved": weather_result.get("location", location),
        "data_timestamp": datetime.now().isoformat()
    }


async def format_for_tts(llm_result: str, context: TaskContext,
                       pre_data: Dict) -> str:
    """
    Post-processor: Format result for text-to-speech.

    Args:
        llm_result: Result from LLM processing
        context: Task context
        pre_data: Data from pre-processing

    Returns:
        TTS-formatted result
    """
    # Replace technical terms with pronunciations
    tts_result = llm_result.replace("°F", " degrees Fahrenheit")
    tts_result = tts_result.replace("mph", " miles per hour")
    tts_result = tts_result.replace("%", " percent")

    return tts_result
```

### Step 3: Test Your Migration

Create tests to verify your hybrid role works correctly:

```python
# tests/unit/test_my_hybrid_role.py

import pytest
from llm_provider.role_registry import RoleRegistry

def test_hybrid_role_loading():
    """Test hybrid role loads correctly."""
    registry = RoleRegistry("roles")

    # Verify execution type
    assert registry.get_role_execution_type("weather") == "hybrid"

    # Verify parameters
    parameters = registry.get_role_parameters("weather")
    assert "location" in parameters
    assert parameters["location"]["required"] is True

    # Verify lifecycle functions
    lifecycle_functions = registry.get_lifecycle_functions("weather")
    assert "fetch_weather_data" in lifecycle_functions
    assert "format_for_tts" in lifecycle_functions
```

## Creating New Hybrid Roles

### 1. Role Definition Template

```yaml
# roles/my_role/definition.yaml

role:
  name: "my_role"
  version: "1.0.0"
  description: "Description of what this role does"
  execution_type: "hybrid"
  fast_reply: true

# Define parameters for routing extraction
parameters:
  required_param:
    type: "string"
    required: true
    description: "Description of required parameter"
    examples: ["example1", "example2"]

  optional_param:
    type: "string"
    required: false
    enum: ["option1", "option2", "option3"]
    default: "option1"

# Configure lifecycle hooks
lifecycle:
  pre_processing:
    enabled: true
    functions:
      - name: "fetch_data"
        uses_parameters: ["required_param", "optional_param"]
      - name: "validate_input"
        uses_parameters: ["required_param"]

  post_processing:
    enabled: true
    functions:
      - name: "format_output"
      - name: "audit_log"

# System prompt with data injection placeholders
prompts:
  system: |
    You are a specialist for {role_purpose}.

    Pre-fetched data:
    - Data: {fetched_data}
    - Validation: {validation_result}

    Use this data to provide helpful responses.

model_config:
  temperature: 0.1
  max_tokens: 2048

tools:
  automatic: false
  shared: [] # Data is pre-fetched
```

### 2. Lifecycle Functions Template

```python
# roles/my_role/lifecycle.py

import asyncio
from datetime import datetime
from typing import Dict, Any
from common.task_context import TaskContext
import logging

logger = logging.getLogger(__name__)


async def fetch_data(instruction: str, context: TaskContext,
                    parameters: Dict) -> Dict[str, Any]:
    """Pre-processor: Fetch required data."""
    required_param = parameters.get("required_param")
    optional_param = parameters.get("optional_param", "default_value")

    # Your data fetching logic here
    data = await your_data_source(required_param, optional_param)

    return {
        "fetched_data": data,
        "fetch_timestamp": datetime.now().isoformat()
    }


async def validate_input(instruction: str, context: TaskContext,
                        parameters: Dict) -> Dict[str, Any]:
    """Pre-processor: Validate input parameters."""
    required_param = parameters.get("required_param")

    # Your validation logic here
    is_valid = validate_parameter(required_param)

    return {
        "validation_result": "valid" if is_valid else "invalid",
        "validated_param": required_param
    }


async def format_output(llm_result: str, context: TaskContext,
                       pre_data: Dict) -> str:
    """Post-processor: Format the output."""
    # Your formatting logic here
    formatted_result = apply_formatting(llm_result)
    return formatted_result


async def audit_log(llm_result: str, context: TaskContext,
                   pre_data: Dict) -> str:
    """Post-processor: Log the interaction."""
    # Your audit logging here
    log_interaction(context, llm_result, pre_data)
    return llm_result  # Return unchanged
```

## Best Practices

### Parameter Design

1. **Keep parameters simple**: Use basic types (string, number, boolean)
2. **Use enums for constrained values**: Helps with validation and routing
3. **Provide good examples**: Helps the routing LLM extract parameters correctly
4. **Set sensible defaults**: For optional parameters

### Pre-processing Functions

1. **Keep functions focused**: Each function should do one thing well
2. **Handle errors gracefully**: Log errors and provide fallback data
3. **Use parameter filtering**: Only pass relevant parameters to each function
4. **Return structured data**: Use consistent key names for LLM injection

### Post-processing Functions

1. **Chain processing**: Each function receives the output of the previous
2. **Preserve original on failure**: Return original result if processing fails
3. **Log processing steps**: For debugging and monitoring
4. **Keep transformations reversible**: Where possible

### System Prompt Design

1. **Reference pre-processed data**: Use `{key_name}` placeholders
2. **Explain data context**: Tell the LLM what data is available
3. **Focus on interpretation**: LLM should interpret, not fetch data
4. **Handle missing data**: Provide fallback instructions

## Common Patterns

### Data Fetching Pattern

```python
async def fetch_external_data(instruction: str, context: TaskContext,
                             parameters: Dict) -> Dict[str, Any]:
    """Fetch data from external API."""
    try:
        # Extract parameters
        key_param = parameters.get("key_param")

        # Fetch data
        api_result = await external_api_call(key_param)

        # Return structured data
        return {
            "api_data": api_result,
            "fetch_timestamp": datetime.now().isoformat(),
            "source": "external_api"
        }
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        raise
```

### Validation Pattern

```python
async def validate_parameters(instruction: str, context: TaskContext,
                             parameters: Dict) -> Dict[str, Any]:
    """Validate and normalize parameters."""
    errors = []
    normalized = {}

    # Validate each parameter
    for param_name, param_value in parameters.items():
        try:
            normalized[param_name] = normalize_parameter(param_value)
        except ValueError as e:
            errors.append(f"{param_name}: {e}")

    return {
        "validation_errors": errors,
        "normalized_parameters": normalized,
        "validation_status": "valid" if not errors else "invalid"
    }
```

### Formatting Pattern

```python
async def format_for_channel(llm_result: str, context: TaskContext,
                            pre_data: Dict) -> str:
    """Format result for specific output channel."""
    # Apply channel-specific formatting
    if context.output_channel == "slack":
        return format_for_slack(llm_result)
    elif context.output_channel == "tts":
        return format_for_tts(llm_result)
    else:
        return llm_result
```

## Troubleshooting

### Common Issues

1. **Lifecycle functions not loading**

   - Check file path: `roles/{role_name}/lifecycle.py`
   - Verify function names match YAML configuration
   - Check for syntax errors in lifecycle.py

2. **Parameters not extracted**

   - Verify parameter schema in role definition
   - Check routing prompt includes parameter information
   - Test with simple, clear parameter requests

3. **Data injection failing**

   - Check placeholder names in system prompt match pre-processing return keys
   - Verify pre-processing functions return dictionaries
   - Handle missing data gracefully in system prompt

4. **Async execution issues**
   - Lifecycle functions must be `async def`
   - Use `await` when calling other async functions
   - Handle event loop properly in tests

### Debugging Tips

1. **Enable debug logging**:

   ```python
   import logging
   logging.getLogger('llm_provider').setLevel(logging.DEBUG)
   ```

2. **Test lifecycle functions independently**:

   ```python
   # Test pre-processing function directly
   result = await fetch_weather_data("test", context, {"location": "Seattle"})
   print(result)
   ```

3. **Verify role loading**:
   ```python
   from llm_provider.role_registry import RoleRegistry
   registry = RoleRegistry("roles")
   print(registry.get_role_execution_type("my_role"))
   print(registry.get_lifecycle_functions("my_role").keys())
   ```

## Performance Considerations

### Routing Performance

- **Token increase**: ~135% (parameter schemas in prompt)
- **Latency increase**: ~135% (more complex routing decision)
- **Offset by**: Eliminated redundant LLM calls for parameter extraction

### Execution Performance

- **Pre-processing**: Runs in parallel where possible
- **Data caching**: Pre-fetched data can be cached
- **Reduced LLM calls**: No tool calls needed for data fetching

### Memory Usage

- **Lifecycle functions**: Loaded once at startup
- **Parameter schemas**: Cached in role registry
- **Pre-processed data**: Temporary, cleaned up after execution

## Migration Checklist

- [ ] Update role definition with `execution_type: "hybrid"`
- [ ] Add parameter schema with types, constraints, and examples
- [ ] Add lifecycle configuration with pre/post processing functions
- [ ] Create `lifecycle.py` file with async functions
- [ ] Update system prompt to reference pre-processed data
- [ ] Remove redundant tools from role configuration
- [ ] Test parameter extraction with various inputs
- [ ] Test pre-processing functions independently
- [ ] Test post-processing functions independently
- [ ] Test end-to-end hybrid execution
- [ ] Verify backward compatibility with existing code
- [ ] Update role documentation and examples

## Examples

See the migrated weather role in `roles/weather/` for a complete example of:

- Parameter schema design
- Lifecycle function implementation
- System prompt with data injection
- Comprehensive test coverage

The weather role demonstrates all hybrid role features:

- Location parameter extraction with multiple formats
- Weather data pre-fetching
- TTS formatting post-processing
- PII scrubbing for security
- Audit logging for compliance
