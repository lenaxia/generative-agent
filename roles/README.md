# Generative Agent Roles Guide

This guide provides comprehensive instructions for creating and adding new roles to the generative agent system. Roles are the fundamental building blocks that define how the system responds to different types of requests, combining language model capabilities with custom pre-processing and post-processing logic.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Creating a New Role](#creating-a-new-role)
- [YAML Schema Reference](#yaml-schema-reference)
- [Lifecycle Functions](#lifecycle-functions)
- [Custom Tools](#custom-tools)
- [Parameter Extraction](#parameter-extraction)
- [Testing Your Role](#testing-your-role)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Directory Structure

The roles system follows a structured directory organization that makes it easy to add new capabilities:

```
roles/
├── role_name/              # Each role has its own directory
│   ├── definition.yaml     # Core configuration that defines the role's behavior
│   ├── tools.py            # Optional custom tools specific to this role
│   └── lifecycle.py        # Pre/post processing functions for enhanced execution
└── shared_tools/           # Tools that can be shared across multiple roles
    ├── calendar_tools.py
    ├── weather_tools.py
    └── ...
```

Each role is self-contained in its own directory, making it easy to add, modify, or remove roles without affecting the rest of the system.

## Creating a New Role

Creating a new role involves several steps, each building upon the previous one to create a complete, functioning component in the system.

### Step 1: Create Directory Structure

First, create a directory for your role. The directory name should match your intended role name:

```bash
mkdir -p roles/my_role_name
```

This directory will contain all the files needed for your role to function.

### Step 2: Create Role Definition

Next, create a `definition.yaml` file that configures how your role behaves. This file is the central blueprint for your role and defines its capabilities, parameters, and execution flow:

```yaml
role:
  name: "weather"
  version: "2.0.0"
  description: "Weather information with pre-processing"
  fast_reply: true  # Enables this role for fast-path routing

# Parameters define what information the role needs from user requests
# These are automatically extracted during routing
parameters:
  location:
    type: "string"
    required: true  # The role cannot function without this parameter
    description: "Location for weather lookup"
    examples: ["Seattle", "New York, NY", "90210"]  # Helps parameter extraction
  
  timeframe:
    type: "string"
    required: false  # Optional parameter with a default value
    enum: ["current", "today", "tomorrow", "this week"]  # Constrains possible values
    default: "current"  # Used when parameter isn't provided

# Lifecycle hooks define the execution flow
lifecycle:
  pre_processing:
    enabled: true  # Enable pre-processing phase
    functions:
      - name: "fetch_weather_data"  # Function must exist in lifecycle.py
        uses_parameters: ["location", "timeframe"]  # Parameters to pass to the function
  
  post_processing:
    enabled: true  # Enable post-processing phase
    functions:
      - name: "format_for_tts"  # Function must exist in lifecycle.py

# System prompt for language model execution
prompts:
  system: |
    You are a weather specialist. Weather data has been pre-fetched:
    Current weather: {weather_current}  # Filled from pre-processing data
    Location: {location_resolved}       # Filled from pre-processing data
    
    Interpret this data naturally for the user.

# Tool configuration
tools:
  automatic: false  # Don't automatically select tools
  shared: []        # No shared tools needed - data is pre-fetched
```

This YAML file defines:
- Basic metadata about your role
- Parameters that will be extracted from user queries
- The execution lifecycle with pre and post-processing hooks
- The system prompt that guides language model behavior
- Tool configuration for your role

The system is very flexible, allowing you to create roles that range from simple Q&A to complex multi-step processes with API integrations.

### Step 3: Implement Custom Tools (Optional)

If your role needs specialized tools beyond the shared ones, create a `tools.py` file in your role directory:

```python
from typing import Dict, Any, List

def tool_name(query: str) -> Dict[str, Any]:
    """
    Tool description here.
    
    Args:
        query: The search query
        
    Returns:
        Dict with search results
    """
    # Tool implementation here
    results = perform_search(query)
    return {"results": results}
```

All functions in this file are automatically loaded and made available to your role without any additional configuration. These tools are specific to your role and won't be available to other roles unless explicitly shared.

The tools follow a standard format:
- They are regular Python functions (not async)
- They include detailed docstrings that explain their purpose and parameters
- They return structured data as dictionaries or other serializable types

Tools should focus on well-defined operations that require external data or computation, like API calls, data processing, or specialized algorithms.

### Step 4: Implement Lifecycle Functions

The most powerful feature of the roles system is the lifecycle hooks. These let you run custom logic before and after language model execution, creating a three-phase process:
1. **Pre-processing**: Fetch data, validate input, transform parameters
2. **Language model execution**: Generate natural language content
3. **Post-processing**: Format output, validate results, log interactions

Create `roles/my_role_name/lifecycle.py` with your lifecycle functions:

```python
import asyncio
from typing import Dict, Any
from common.task_context import TaskContext

async def fetch_data(instruction: str, context: TaskContext, 
                    parameters: Dict) -> Dict[str, Any]:
    """
    Pre-processor: Fetch required data.
    
    Args:
        instruction: Original user instruction
        context: Task context
        parameters: Extracted parameters from routing
        
    Returns:
        Dict containing data for LLM context
    """
    # Your data fetching logic here
    # This function runs BEFORE the language model
    # Results are injected into the system prompt
    param = parameters.get("required_param")
    return {"fetched_data": f"Data for {param}"}

async def format_output(llm_result: str, context: TaskContext, 
                       pre_data: Dict) -> str:
    """
    Post-processor: Format the output.
    
    Args:
        llm_result: Result from LLM processing
        context: Task context
        pre_data: Data from pre-processing
        
    Returns:
        Formatted result
    """
    # Your formatting logic here
    # This function runs AFTER the language model
    # It can transform the result before returning to the user
    return llm_result.replace("technical_term", "simple explanation")
```

Lifecycle functions follow these rules:
- They must be `async` functions (use `async def`)
- Pre-processors receive parameters and return a dictionary of data
- Post-processors receive the language model output and pre-processor data
- They are executed in the order defined in the YAML file
- If a pre-processor fails, the execution falls back to using just the language model

This lifecycle approach separates concerns cleanly:
- Pre-processors focus on data gathering and preparation
- The language model focuses on interpretation and generation
- Post-processors focus on formatting and standardization

## YAML Schema Reference

The role definition YAML file has several sections, each controlling a different aspect of the role's behavior.

### Role Definition

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| role.name | string | Yes | Role name (must match directory name) |
| role.version | string | Yes | Semantic version (e.g., "1.0.0") |
| role.description | string | Yes | Brief description of the role |
| role.fast_reply | boolean | No | Enable for fast-path routing (default: false). When true, this role can be selected directly without complex planning. |
| role.when_to_use | string | No | Guidance on when to use this role. Helps the system make better routing decisions. |

### Parameters

Parameters are automatically extracted from user queries during routing and made available to your pre-processing functions:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| parameters.{name}.type | string | Yes | Parameter type ("string", "number", "boolean") |
| parameters.{name}.required | boolean | Yes | Whether parameter is required for the role to function |
| parameters.{name}.description | string | Yes | Description of parameter purpose (helps extraction) |
| parameters.{name}.examples | array | No | Example valid values (improves extraction accuracy) |
| parameters.{name}.enum | array | No | Allowed values for constrained parameters (enforces validation) |
| parameters.{name}.default | * | No | Default value for optional parameters |

Adding good examples and clear descriptions helps the system extract parameters more accurately from natural language queries.

### Lifecycle

Lifecycle configuration defines the execution flow of your role:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| lifecycle.pre_processing.enabled | boolean | Yes | Enable pre-processing phase |
| lifecycle.pre_processing.functions | array | Yes | List of pre-processor functions to run before language model execution |
| lifecycle.pre_processing.functions[].name | string | Yes | Function name (must exist in lifecycle.py) |
| lifecycle.pre_processing.functions[].uses_parameters | array | No | Parameters used by this function (filtered from extracted parameters) |
| lifecycle.post_processing.enabled | boolean | Yes | Enable post-processing phase |
| lifecycle.post_processing.functions | array | Yes | List of post-processor functions to run after language model execution |

Pre-processing functions run before the language model and can inject data into the context. Post-processing functions run after and can modify the output.

### Prompts

The prompts section defines how the language model behaves:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| prompts.system | string | Yes | System prompt for the language model. Can include {placeholders} for pre-processed data. |

The system prompt is the foundation of your role's behavior. It can include placeholders like `{weather_data}` that will be filled in with data from pre-processing functions.

### Tools

Tools configuration determines what tools are available to your role:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| tools.automatic | boolean | No | Enable automatic tool selection based on the task (default: false) |
| tools.shared | array | No | List of shared tools to include from shared_tools directory |

Custom tools from your role's `tools.py` are always included automatically. The `shared` list lets you include tools from the shared directory.

### Model Configuration

Fine-tune language model behavior with these settings:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| model_config.temperature | number | No | Temperature for language model output (0.0 - 1.0). Lower values produce more deterministic outputs. |
| model_config.max_tokens | number | No | Maximum tokens in response |
| model_config.max_context | number | No | Maximum context size |
| model_config.top_p | number | No | Top-p sampling parameter |

Lower temperature values (0.1 - 0.3) are better for factual tasks, while higher values (0.6 - 0.9) allow more creativity.

## Lifecycle Functions

The lifecycle architecture is one of the most powerful features of the roles system. It divides task execution into three phases:

### Pre-processors

Pre-processors run before language model execution and provide data for the context. They're ideal for:
- Fetching data from APIs
- Validating user input
- Transforming parameters into useful formats
- Retrieving content from databases or files
- Performing complex calculations

```python
async def function_name(instruction: str, context: TaskContext, 
                       parameters: Dict) -> Dict[str, Any]:
    """
    Pre-processor function documentation.
    
    Args:
        instruction: Original user instruction
        context: Task context
        parameters: Extracted parameters from routing
        
    Returns:
        Dict with data to inject into LLM context
    """
    # Implementation
    return {"key1": "value1", "key2": "value2"}
```

The keys in the returned dictionary can be referenced in the system prompt using `{key1}` and `{key2}` placeholders. All pre-processor results are combined into a single dictionary for the language model.

If a pre-processor fails (raises an exception), the system will log the error and continue execution without the data.

### Post-processors

Post-processors run after language model execution and can modify the output:

```python
async def function_name(llm_result: str, context: TaskContext, 
                       pre_data: Dict) -> str:
    """
    Post-processor function documentation.
    
    Args:
        llm_result: Output from language model
        context: Task context
        pre_data: Combined data from all pre-processors
        
    Returns:
        Modified language model output
    """
    # Implementation
    return modified_result
```

Post-processors are ideal for:
- Formatting output for specific channels (Slack, email, TTS)
- Removing sensitive information
- Adding citations or references
- Validating language model output
- Logging interactions for compliance
- Adding metadata for downstream systems

Post-processors run in sequence, with each receiving the output from the previous one. If a post-processor fails, the system returns the most recent successful output.

## Custom Tools

Custom tools extend what your role can do during language model execution. Unlike lifecycle functions, tools are called *by* the language model when it needs specific information or actions.

Tools follow this pattern:

```python
def tool_name(parameter1: str, parameter2: int = 10) -> Dict[str, Any]:
    """
    Tool description that explains when and how to use this tool.
    
    Args:
        parameter1: First parameter description
        parameter2: Second parameter description
        
    Returns:
        Description of return value
    """
    # Tool implementation
    return {"result": "some result"}
```

Key characteristics of tools:
- They are synchronous functions (not async)
- They have descriptive docstrings that guide the model
- They return structured data (dictionaries, lists, primitives)
- They should be focused on a single task
- They should handle errors gracefully

All functions in your role's `tools.py` file are automatically loaded and made available to your role. You don't need to list them in the YAML configuration.

### Shared Tools

The shared_tools directory contains tools that can be used by multiple roles. This avoids code duplication and ensures consistent behavior across roles.

To use shared tools:

1. Create a file in `roles/shared_tools/` (e.g., `roles/shared_tools/my_shared_tools.py`)
2. Implement your tool functions
3. Reference the tool name in your role's `tools.shared` list in definition.yaml

For example, if you have a shared tool in `roles/shared_tools/search_tools.py` named `web_search`, you would add:

```yaml
tools:
  shared: ["web_search"]
```

This makes the shared tool available to your role without duplicating code. If the same functionality is needed in multiple roles, it should be a shared tool.

## Parameter Extraction

The parameter extraction system automatically pulls structured data from natural language queries:

1. Define parameters in your role definition YAML
2. The router extracts these parameters when the role is selected
3. Parameters are passed to your pre-processing functions

For example, with this parameter definition:

```yaml
parameters:
  location:
    type: "string"
    required: true
    description: "City, state, country, or coordinates"
    examples: ["Seattle", "New York, NY", "90210"]
```

And this user query:
> "What's the weather like in Seattle tomorrow?"

The system will extract:
```json
{
  "location": "Seattle",
  "timeframe": "tomorrow"
}
```

These parameters are then available in your pre-processing functions. You can specify which parameters each function needs with `uses_parameters`:

```yaml
lifecycle:
  pre_processing:
    enabled: true
    functions:
      - name: "fetch_weather_data"
        uses_parameters: ["location", "timeframe"]
```

This ensures your functions only receive relevant parameters, even if the system extracts additional ones.

The parameter extraction system handles a wide range of natural language variations and can match parameters even when they're expressed differently than your examples.

## Testing Your Role

Good testing ensures your role works correctly and continues to work as the system evolves.

Create tests to verify your role's behavior:

```python
# tests/unit/test_my_role.py

import pytest
from llm_provider.role_registry import RoleRegistry

def test_role_loading():
    """Test role loads correctly."""
    registry = RoleRegistry("roles")
    assert registry.get_role("my_role") is not None

def test_lifecycle_functions():
    """Test lifecycle functions are registered."""
    registry = RoleRegistry("roles")
    functions = registry.get_lifecycle_functions("my_role")
    assert "fetch_data" in functions
    assert "format_output" in functions

def test_pre_processing():
    """Test pre-processing functions."""
    # Implement pre-processing function test
    pass

def test_post_processing():
    """Test post-processing functions."""
    # Implement post-processing function test
    pass
```

Testing should cover:
1. **Role loading**: The role is properly discovered and loaded
2. **Lifecycle functions**: Pre and post-processors are correctly registered
3. **Parameter extraction**: Parameters are correctly pulled from queries
4. **Pre-processing logic**: Data is fetched and formatted correctly
5. **Post-processing logic**: Output is transformed as expected
6. **End-to-end flow**: The complete role behaves as expected

Use mock objects for external dependencies (APIs, databases) to make tests reliable and fast.

## Best Practices

### Role Design

1. **Single Responsibility**: Each role should do one thing well. If a role is trying to do too many things, consider splitting it into multiple roles.

2. **Clear Parameters**: Define clear parameter schemas with examples. Good parameter definitions improve extraction accuracy and make your role more predictable.

3. **Pre-fetch Data**: Use pre-processing for data fetching, validation, and transformation. This ensures your language model has all the data it needs before generating a response.

4. **Focus on Interpretation**: The system should interpret data, not fetch it repeatedly. Pre-processing should handle data gathering, while the language model focuses on explaining and interpreting that data.

5. **Post-process for Consistency**: Ensure consistent output formatting with post-processors. This is especially important for multi-channel interfaces (web, chat, voice) or when outputs need to meet specific format requirements.

### Parameter Design

1. **Keep parameters simple**: Use basic types (string, number, boolean). Complex nested structures are harder to extract accurately.

2. **Use enums for constrained values**: Define allowed options for parameters with limited valid values. This improves extraction and prevents invalid inputs.

3. **Provide good examples**: Include diverse, realistic examples that cover common ways users might express the parameter. This improves extraction accuracy, especially for edge cases.

4. **Set sensible defaults**: For optional parameters, always provide defaults that make sense for most users. This allows your role to function even when parameters aren't explicitly mentioned.

### Performance Optimization

1. **Pre-fetch heavy data**: Use pre-processing for network calls and databases. This reduces the latency experienced by users and avoids timeout issues.

2. **Cache where possible**: Cache expensive operations, especially for data that doesn't change frequently. This improves response times for repeat queries.

3. **Keep pre/post processing lightweight**: Minimize processing time in lifecycle functions. Heavy computation should be optimized or moved to background processes when possible.

4. **Use appropriate model configurations**: Match model to task complexity. Simpler tasks can use smaller, faster models, while complex tasks may need more powerful models.

## Troubleshooting

### Common Issues

1. **Role not loading**
   - Check directory structure matches role name exactly (case-sensitive)
   - Verify YAML syntax is correct (no missing colons, proper indentation)
   - Check for required fields in definition (name, version, description)
   - Look at error logs for specific parsing errors

2. **Lifecycle functions not found**
   - Verify function names match exactly between YAML and lifecycle.py
   - Check for syntax errors in lifecycle.py
   - Ensure functions are properly defined as `async def`
   - Check import statements for required modules

3. **Parameter extraction issues**
   - Add more diverse examples to parameters
   - Check parameter types and constraints
   - Test with clear, unambiguous queries
   - Review routing logs to see what's being extracted

4. **Data injection failing**
   - Ensure pre-processing function return keys match placeholders in system prompt
   - Check for properly formatted dictionaries in return values
   - Verify pre-processing functions aren't raising exceptions
   - Look for typos in placeholder names (they're case-sensitive)

### Debugging Tips

1. **Enable debug logging**
   ```python
   import logging
   logging.getLogger('llm_provider').setLevel(logging.DEBUG)
   ```

   This shows detailed logs of role loading, parameter extraction, and lifecycle execution.

2. **Test roles individually**
   ```python
   from llm_provider.role_registry import RoleRegistry
   registry = RoleRegistry("roles")
   role = registry.get_role("my_role")
   print(role.config)
   ```

   This lets you inspect a role's configuration outside of the normal execution flow.

3. **Test lifecycle functions directly**
   ```python
   import asyncio
   from roles.my_role.lifecycle import fetch_data
   result = asyncio.run(fetch_data("test instruction", None, {"param": "value"}))
   print(result)
   ```

   This bypasses the router and executes lifecycle functions directly for debugging.

When debugging, focus on isolating the problem:
- Is it the role definition (YAML)?
- Is it the lifecycle functions (Python)?
- Is it parameter extraction?
- Is it the system prompt?

## Examples

See these example roles for reference:

1. **Weather Role**: `roles/weather/` - Complete reference implementation with:
   - Parameter extraction (location, timeframe, format)
   - Pre-processing to fetch weather data from API
   - Post-processing for TTS formatting, PII scrubbing, and audit logging
   - Demonstrates all hybrid role lifecycle features

2. **Search Role**: `roles/search/` - Web search with API integration and result formatting

3. **Calendar Role**: `roles/calendar/` - Calendar integration with date/time handling

4. **Timer Role**: `roles/timer/` - Timer management and scheduling

5. **Planning Role**: `roles/planning/` - Complex task planning and dependency analysis

All roles use the unified hybrid architecture for consistent behavior, performance, and monitoring.

For a complete demonstration of the role architecture, see:
- `examples/hybrid_role_example.py` - Working demonstration
- `docs/HYBRID_ROLE_MIGRATION_GUIDE.md` - Step-by-step creation guide

## Additional Resources

- [Tool Development Guide](../docs/05_TOOL_DEVELOPMENT_GUIDE.md) - Guide for creating new tools
- [API Reference](../docs/02_API_REFERENCE.md) - Full API documentation
