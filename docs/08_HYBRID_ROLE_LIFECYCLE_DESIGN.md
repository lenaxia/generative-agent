# Hybrid Role Lifecycle Architecture Design

## Rules

- Always use the venv at ./venv/bin/activate
- ALWAYS use test driven development, write tests first
- Never assume tests pass, run the tests and positively verify that the test passed
- ALWAYS run all tests after making any change to ensure they are still all passing, do not move on until relevant tests are passing
- If a test fails, reflect deeply about why the test failed and fix it or fix the code
- Always write multiple tests, including happy, unhappy path and corner cases
- Always verify interfaces and data structures before writing code, do not assume the definition of a interface or data structure
- When performing refactors, ALWAYS use grep to find all instances that need to be refactored
- If you are stuck in a debugging cycle and can't seem to make forward progress, either ask for user input or take a step back and reflect on the broader scope of the code you're working on
- ALWAYS make sure your tests are meaningful, do not mock excessively, only mock where ABSOLUTELY necessary.
- Make a git commit after major changes have been completed
- When refactoring an object, refactor it in place, do not create a new file just for the sake of preserving the old version, we have git for that reason. For instance, if refactoring RequestManager, do NOT create an EnhancedRequestManager, just refactor or rewrite RequestManager
- ALWAYS Follow development and language best practices
- Use the Context7 MCP server if you need documentation for something, make sure you're looking at the right version
- Remember we are migrating AWAY from langchain TO strands agent
- Do not worry about backwards compatibility unless it is PART of a migration process and you will remove the backwards compatibility later
- Do not use fallbacks
- Whenever you complete a phase, make sure to update this checklist
- Don't just blindly implement changes. Reflect on them to make sure they make sense within the larger project. Pull in other files if additional context is needed
- Always use in place refactoring, i.e. instead of creating a EhnancedVersionOfObject just refactor Object as is. Only create new objects when it actually makes sense to.

## Overview

This document outlines the design for implementing hybrid roles with lifecycle hooks (pre-processing and post-processing) and enhanced routing with parameter extraction. This architecture unifies the current dual role system (LLM + Programmatic) into a single, configurable hybrid execution model using **in-place refactoring** of existing components.

## Current Architecture

### Existing Components

- **WorkflowEngine**: Manages DAG-based task execution
- **RoleRegistry**: Supports both LLM and programmatic roles
- **UniversalAgent**: Executes tasks using different role types
- **RequestRouter**: Routes requests to fast-reply roles or complex workflows
- **Two Role Types**:
  - **LLM Roles**: YAML-defined with system prompts and tools
  - **Programmatic Roles**: Python classes that execute directly

### Current Execution Flow

```
Request → Router → Role Selection → Execution → Result
```

## Proposed Hybrid Architecture

### New Execution Flow

```
Request → Enhanced Router (with parameter extraction) → Hybrid Role Execution → Result
                                                           ↓
                                                    Pre-processing
                                                           ↓
                                                    LLM Processing (optional)
                                                           ↓
                                                    Post-processing
```

## Enhanced Role Definition Schema

### Enhanced Parameter Definition

```yaml
# Enhanced role definition with lifecycle hooks
role:
  name: "weather"
  version: "2.0.0"
  description: "Weather role with pre-processing data fetching"
  fast_reply: true

# Enhanced parameter schema for routing extraction
parameters:
  location:
    type: "string"
    required: true
    description: "City, state, country, or coordinates for weather lookup"
    examples: ["Seattle", "New York, NY", "90210", "47.6062,-122.3321"]

  timeframe:
    type: "string"
    required: false
    description: "When to get weather for"
    examples: ["current", "today", "tomorrow", "this week"]
    enum: ["current", "today", "tomorrow", "this week", "next week"] # Optional: restrict to specific values
    default: "current"

  format:
    type: "string"
    required: false
    description: "Output format preference"
    enum: ["brief", "detailed", "forecast"] # LLM must pick from this list
    default: "brief"

# Lifecycle hooks for hybrid execution
lifecycle:
  pre_processing:
    enabled: true
    functions:
      - name: "fetch_weather_data"
        uses_parameters: ["location", "timeframe"]
      - name: "validate_location"
        uses_parameters: ["location"]
    data_injection: true # Inject results into LLM context

  post_processing:
    enabled: true
    functions:
      - name: "pii_scrubber"
        description: "Remove sensitive data"
      - name: "format_for_tts"
        description: "Format response for text-to-speech"
      - name: "audit_log"
        description: "Log interaction for compliance"

# Traditional LLM configuration (enhanced with pre-processed data)
prompts:
  system: |
    You are a weather specialist. The weather data has already been fetched
    and is available in your context as {weather_data}. Focus on interpreting
    and explaining this data rather than fetching it.

    Available data:
    - Current weather: {weather_current}
    - Location resolved: {location_resolved}
    - Data timestamp: {data_timestamp}

# Model and tool configuration remains the same
model_config:
  temperature: 0.1
  max_tokens: 2048

tools:
  automatic: false
  shared: [] # No need for weather tools since data is pre-fetched
```

## Component Modifications (In-Place Refactoring)

### 1. RoleRegistry Enhancements

#### Enhanced Methods (Added to Existing Class)

```python
# In llm_provider/role_registry.py

class RoleRegistry:
    def __init__(self, roles_directory: str = "roles"):
        # Existing initialization...
        # Add lifecycle function storage for hybrid roles
        self.lifecycle_functions: Dict[str, Dict[str, Callable]] = {}  # role_name -> {func_name: func}

    def get_role_parameters(self, role_name: str) -> Dict[str, Any]:
        """Get parameter schema for a role for routing extraction."""
        role_def = self.get_role(role_name)
        if not role_def:
            return {}
        return role_def.config.get('parameters', {})

    def register_lifecycle_functions(self, role_name: str, functions: Dict[str, Callable]):
        """Register lifecycle functions for a role."""
        self.lifecycle_functions[role_name] = functions
        logger.info(f"Registered {len(functions)} lifecycle functions for role: {role_name}")

    def get_lifecycle_functions(self, role_name: str) -> Dict[str, Callable]:
        """Get lifecycle functions for a role."""
        return self.lifecycle_functions.get(role_name, {})

    def _load_role(self, role_name: str) -> RoleDefinition:
        """Enhanced role loading with lifecycle function support."""
        # Existing role loading logic...
        role_def = RoleDefinition(...)

        # Load lifecycle functions for all roles
        lifecycle_functions = self._load_lifecycle_functions(role_name)
        if lifecycle_functions:
            self.register_lifecycle_functions(role_name, lifecycle_functions)

        return role_def

    def _load_lifecycle_functions(self, role_name: str) -> Dict[str, Callable]:
        """Load lifecycle functions from role's Python module."""
        # Load from roles/{role_name}/lifecycle.py if it exists
        lifecycle_file = self.roles_directory / role_name / "lifecycle.py"
        if lifecycle_file.exists():
            return self._load_functions_from_file(lifecycle_file)
        return {}
```

### 2. RequestRouter Enhancements

#### Enhanced Routing with Parameter Extraction (In-Place)

````python
# In llm_provider/request_router.py

class RequestRouter:
    def route_request(self, instruction: str) -> Dict[str, Any]:
        """
        Enhanced routing with parameter extraction in single LLM call.

        Returns:
            Dict containing route, confidence, and extracted parameters
        """
        import time
        start_time = time.time()

        # Handle None or empty instruction
        if instruction is None:
            return {
                "route": "PLANNING",
                "confidence": 0.0,
                "parameters": {},
                "error": "No instruction provided"
            }

        try:
            # Check cache first
            cache_key = self._create_cache_key(instruction)
            if cache_key in self._routing_cache:
                cached_result = self._routing_cache[cache_key].copy()
                cached_result["execution_time_ms"] = 0.1
                return cached_result

            # Get fast-reply roles with parameter schemas
            fast_reply_roles = self.role_registry.get_fast_reply_roles()
            if not fast_reply_roles:
                return {"route": "PLANNING", "confidence": 0.0, "parameters": {}}

            # Build enhanced routing prompt with parameter schemas
            routing_prompt = self._build_enhanced_routing_prompt(instruction, fast_reply_roles)

            # Single LLM call for routing AND parameter extraction
            result = self.universal_agent.execute_task(
                instruction=routing_prompt,
                role="router",
                llm_type=LLMType.WEAK
            )

            # Parse routing result with parameters
            parsed_result = self._parse_routing_and_parameters(result)

            # Cache the result
            if len(self._routing_cache) >= self._cache_max_size:
                self._routing_cache.clear()
            self._routing_cache[cache_key] = parsed_result.copy()

            execution_time_ms = (time.time() - start_time) * 1000
            parsed_result["execution_time_ms"] = execution_time_ms

            logger.info(f"Enhanced routing: '{parsed_result['route']}' with confidence {parsed_result['confidence']:.2f} and {len(parsed_result.get('parameters', {}))} parameters")
            return parsed_result

        except Exception as e:
            logger.error(f"Enhanced routing failed: {e}")
            return {
                "route": "PLANNING",
                "confidence": 0.0,
                "parameters": {},
                "error": str(e)
            }

    def _build_enhanced_routing_prompt(self, instruction: str, fast_reply_roles: List) -> str:
        """Build routing prompt that includes parameter extraction."""

        # Build role schemas for parameter extraction
        role_schemas = {}
        for role_def in fast_reply_roles:
            role_name = role_def.name
            parameters = self.role_registry.get_role_parameters(role_name)
            if parameters:
                role_schemas[role_name] = {
                    "description": role_def.config.get('role', {}).get('description', ''),
                    "parameters": parameters
                }

        return f"""Route this request to the best role AND extract parameters for that role.

Request: "{instruction}"

Available roles and their parameters:
{json.dumps(role_schemas, indent=2)}

Analyze the request and respond with JSON in this exact format:
{{
  "route": "role_name",
  "confidence": 0.95,
  "parameters": {{
    "param_name": "extracted_value"
  }}
}}

Rules:
- Choose the role that best matches the request intent
- Extract only the parameters defined for the chosen role
- For enum parameters, pick from the allowed values only
- Use examples as guidance for parameter format
- Use confidence 0.0-1.0 based on how well the request matches the role
- If no role matches well, use "PLANNING" with confidence < 0.7
- Ensure all required parameters are extracted if possible"""

    def _parse_routing_and_parameters(self, llm_result: str) -> Dict[str, Any]:
        """Parse LLM result to extract route and parameters."""
        try:
            # Clean the result and parse JSON
            cleaned_result = llm_result.strip()
            if cleaned_result.startswith('```json'):
                cleaned_result = cleaned_result[7:-3].strip()
            elif cleaned_result.startswith('```'):
                cleaned_result = cleaned_result[3:-3].strip()

            parsed = json.loads(cleaned_result)

            return {
                "route": parsed.get("route", "PLANNING"),
                "confidence": float(parsed.get("confidence", 0.0)),
                "parameters": parsed.get("parameters", {})
            }

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse routing result: {e}")
            return {
                "route": "PLANNING",
                "confidence": 0.0,
                "parameters": {},
                "error": f"Parse error: {str(e)}"
            }
````

### 3. UniversalAgent Enhancements

#### Enhanced Task Execution with Lifecycle Support (In-Place)

```python
# In llm_provider/universal_agent.py

class UniversalAgent:
    async def execute_task(self, instruction: str, role: str = "default",
                          llm_type: LLMType = LLMType.DEFAULT,
                          context: Optional[TaskContext] = None,
                          extracted_parameters: Optional[Dict] = None) -> str:
        """
        Enhanced task execution with hybrid role lifecycle support.

        Args:
            instruction: Task instruction
            role: Agent role to assume
            llm_type: Model type for optimization
            context: Optional task context
            extracted_parameters: Parameters extracted during routing

        Returns:
            str: Task result
        """
        # Check execution type
        execution_type = self.role_registry.get_role_execution_type(role)

        if execution_type == "hybrid":
            return await self._execute_hybrid_task(instruction, role, context, extracted_parameters)
        elif execution_type == "programmatic":
            return self._execute_programmatic_task(instruction, role, context)
        else:
            return self._execute_llm_task(instruction, role, llm_type, context)

    async def _execute_hybrid_task(self, instruction: str, role: str,
                                  context: Optional[TaskContext],
                                  extracted_parameters: Optional[Dict]) -> str:
        """Execute hybrid role with lifecycle hooks."""
        start_time = time.time()

        try:
            role_def = self.role_registry.get_role(role)
            if not role_def:
                raise ValueError(f"Role '{role}' not found")

            lifecycle_functions = self.role_registry.get_lifecycle_functions(role)

            # 1. Pre-processing phase
            pre_data = {}
            if self._has_pre_processing(role_def):
                logger.info(f"Running pre-processing for {role}")
                pre_data = await self._run_pre_processors(
                    role_def, lifecycle_functions, instruction, context, extracted_parameters or {}
                )

            # 2. LLM execution phase (if needed)
            llm_result = None
            if self._needs_llm_processing(role_def):
                logger.info(f"Running LLM processing for {role}")
                enhanced_instruction = self._inject_pre_data(role_def, instruction, pre_data)
                llm_result = self._execute_llm_with_context(enhanced_instruction, role, context)

            # 3. Post-processing phase
            final_result = llm_result or self._format_pre_data_result(pre_data)
            if self._has_post_processing(role_def):
                logger.info(f"Running post-processing for {role}")
                final_result = await self._run_post_processors(
                    role_def, lifecycle_functions, final_result, context, pre_data
                )

            execution_time = time.time() - start_time
            logger.info(f"Hybrid role {role} completed in {execution_time:.3f}s")
            return final_result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Hybrid role {role} failed after {execution_time:.3f}s: {e}")
            return f"Error in {role}: {str(e)}"

    async def _run_pre_processors(self, role_def: RoleDefinition, lifecycle_functions: Dict,
                                 instruction: str, context: TaskContext, parameters: Dict) -> Dict[str, Any]:
        """Run all pre-processing functions for a role."""
        results = {}

        pre_config = role_def.config.get('lifecycle', {}).get('pre_processing', {})
        functions = pre_config.get('functions', [])

        for func_config in functions:
            if isinstance(func_config, str):
                func_name = func_config
                func_params = []
            else:
                func_name = func_config.get('name')
                func_params = func_config.get('uses_parameters', [])

            processor = lifecycle_functions.get(func_name)
            if processor:
                try:
                    # Extract relevant parameters for this function
                    func_parameters = {k: v for k, v in parameters.items() if k in func_params}

                    result = await processor(instruction, context, func_parameters)
                    results[func_name] = result
                    logger.debug(f"Pre-processor '{func_name}' completed successfully")

                except Exception as e:
                    logger.error(f"Pre-processor '{func_name}' failed: {e}")
                    results[func_name] = {"error": str(e)}
            else:
                logger.warning(f"Pre-processor function '{func_name}' not found")

        return results

    async def _run_post_processors(self, role_def: RoleDefinition, lifecycle_functions: Dict,
                                  llm_result: str, context: TaskContext, pre_data: Dict) -> str:
        """Run all post-processing functions for a role."""
        current_result = llm_result

        post_config = role_def.config.get('lifecycle', {}).get('post_processing', {})
        functions = post_config.get('functions', [])

        for func_config in functions:
            func_name = func_config if isinstance(func_config, str) else func_config.get('name')

            processor = lifecycle_functions.get(func_name)
            if processor:
                try:
                    current_result = await processor(current_result, context, pre_data)
                    logger.debug(f"Post-processor '{func_name}' completed successfully")

                except Exception as e:
                    logger.error(f"Post-processor '{func_name}' failed: {e}")
                    # Continue with current result on post-processor failure
            else:
                logger.warning(f"Post-processor function '{func_name}' not found")

        return current_result

    def _has_pre_processing(self, role_def: RoleDefinition) -> bool:
        """Check if pre-processing is enabled for a role."""
        return role_def.config.get('lifecycle', {}).get('pre_processing', {}).get('enabled', False)

    def _has_post_processing(self, role_def: RoleDefinition) -> bool:
        """Check if post-processing is enabled for a role."""
        return role_def.config.get('lifecycle', {}).get('post_processing', {}).get('enabled', False)

    def _needs_llm_processing(self, role_def: RoleDefinition) -> bool:
        """Determine if LLM processing is needed for a role."""
        execution_type = role_def.config.get('role', {}).get('execution_type', 'hybrid')
        return execution_type in ['hybrid', 'llm']

    def _inject_pre_data(self, role_def: RoleDefinition, instruction: str, pre_data: Dict) -> str:
        """Inject pre-processing data into instruction context."""
        system_prompt = role_def.config.get('prompts', {}).get('system', '')

        # Format system prompt with pre-processed data
        try:
            formatted_prompt = system_prompt.format(**self._flatten_pre_data(pre_data))
            return f"{formatted_prompt}\n\nUser Request: {instruction}"
        except KeyError as e:
            logger.warning(f"Failed to format system prompt with pre-data: {e}")
            return instruction

    def _flatten_pre_data(self, pre_data: Dict) -> Dict[str, Any]:
        """Flatten pre-processing data for prompt formatting."""
        flattened = {}
        for func_name, data in pre_data.items():
            if isinstance(data, dict) and 'error' not in data:
                # Flatten successful results
                for key, value in data.items():
                    flattened[key] = value
        return flattened

    def _format_pre_data_result(self, pre_data: Dict) -> str:
        """Format pre-processing data as final result (for programmatic-only execution)."""
        return str(pre_data)
```

### 4. WorkflowEngine Integration (In-Place)

#### Enhanced Task Execution (Modified Existing Method)

```python
# In supervisor/workflow_engine.py

class WorkflowEngine:
    def _handle_fast_reply(self, request: RequestMetadata, routing_result: Dict) -> str:
        """Execute fast-reply with hybrid role support and pre-extracted parameters."""
        try:
            request_id = 'fr_' + str(uuid.uuid4()).split('-')[-1]
            role = routing_result["route"]
            parameters = routing_result.get("parameters", {})

            logger.info(f"Fast-reply '{request_id}' via {role} role with {len(parameters)} parameters")

            # Start duration tracking
            duration_logger = get_duration_logger()
            duration_logger.start_workflow_tracking(
                workflow_id=request_id,
                source=WorkflowSource.CLI,
                workflow_type=WorkflowType.FAST_REPLY,
                instruction=request.prompt
            )

            # Check if this is a hybrid role
            execution_type = self.role_registry.get_role_execution_type(role)

            if execution_type == "hybrid":
                # Execute hybrid role with pre-extracted parameters
                result = await self.universal_agent.execute_task(
                    instruction=request.prompt,
                    role=role,
                    llm_type=LLMType.WEAK,
                    context=None,
                    extracted_parameters=parameters
                )
            else:
                # Existing LLM execution path with parameter context injection
                if parameters:
                    param_context = "Context: " + ", ".join([f"{k}={v}" for k, v in parameters.items()])
                    enhanced_instruction = f"{param_context}\n\n{request.prompt}"
                else:
                    enhanced_instruction = request.prompt

                result = self.universal_agent.execute_task(
                    instruction=enhanced_instruction,
                    role=role,
                    llm_type=LLMType.WEAK,
                    context=None
                )

            # Complete duration tracking
            duration_logger.complete_workflow_tracking(
                workflow_id=request_id,
                success=True,
                role=role,
                confidence=routing_result.get('confidence')
            )

            # Store result with parameters
            self._store_fast_reply_result(
                request_id,
                result,
                role=role,
                confidence=routing_result.get('confidence'),
                parameters=parameters
            )

            return request_id

        except Exception as e:
            logger.error(f"Fast-reply execution failed: {e}")
            return self._handle_complex_workflow(request)

    def _store_fast_reply_result(self, request_id: str, result: str, role: str = None,
                                confidence: float = None, parameters: Dict = None,
                                execution_time_ms: float = None):
        """Enhanced result storage with parameters."""
        self.fast_reply_results[request_id] = {
            "result": result,
            "role": role,
            "confidence": confidence,
            "parameters": parameters or {},
            "execution_time_ms": execution_time_ms,
            "timestamp": time.time()
        }
```

## Example Implementation: Weather Role Enhancement

### Enhanced Weather Role Definition

```yaml
# roles/weather/definition.yaml (Enhanced In-Place)
role:
  name: "weather"
  version: "2.0.0"
  description: "Weather role with pre-processing data fetching"
  execution_type: "hybrid" # Changed from "llm" to "hybrid"
  fast_reply: true

# New parameter schema
parameters:
  location:
    type: "string"
    required: true
    description: "City, state, country, or coordinates for weather lookup"
    examples: ["Seattle", "New York, NY", "90210", "47.6062,-122.3321"]

  timeframe:
    type: "string"
    required: false
    description: "When to get weather for"
    enum: ["current", "today", "tomorrow", "this week", "next week"]
    default: "current"

  format:
    type: "string"
    required: false
    description: "Output format preference"
    enum: ["brief", "detailed", "forecast"]
    default: "brief"

# New lifecycle configuration
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

# Enhanced system prompt
prompts:
  system: |
    You are a weather specialist. Weather data has been pre-fetched for you:

    Current Weather: {weather_current}
    Location: {location_resolved}
    Timestamp: {data_timestamp}

    Interpret and explain this weather data in a natural, conversational way.
    Focus on what the user needs to know about the weather conditions.

model_config:
  temperature: 0.1
  max_tokens: 2048

tools:
  automatic: false
  shared: [] # No weather tools needed - data is pre-fetched
```

### Weather Lifecycle Functions

```python
# roles/weather/lifecycle.py (New File)

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from roles.shared_tools.weather_tools import get_weather, get_weather_forecast
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
        parameters: Extracted parameters (location, timeframe)

    Returns:
        Dict containing weather data for LLM context
    """
    location = parameters.get("location")
    timeframe = parameters.get("timeframe", "current")

    if not location:
        raise ValueError("Location parameter is required for weather data")

    try:
        # Fetch current weather data
        if timeframe in ["current", "now", "today"]:
            weather_result = get_weather(location)
        else:
            # For future timeframes, get forecast
            weather_result = get_weather_forecast(location, days=7)

        if weather_result.get("status") == "error":
            raise ValueError(f"Weather API error: {weather_result.get('error')}")

        return {
            "weather_current": weather_result.get("weather", {}),
            "location_resolved": weather_result.get("location", location),
            "coordinates": weather_result.get("coordinates", {}),
            "data_timestamp": datetime.now().isoformat(),
            "timeframe_requested": timeframe
        }

    except Exception as e:
        logger.error(f"Failed to fetch weather data for {location}: {e}")
        raise


async def format_for_tts(llm_result: str, context: TaskContext,
                       pre_data: Dict) -> str:
    """
    Post-processor: Format LLM result for text-to-speech.

    Args:
        llm_result: Result from LLM processing
        context: Task context
        pre_data: Data from pre-processing

    Returns:
        TTS-formatted result
    """
    try:
        # Remove markdown formatting
        tts_result = llm_result.replace("**", "").replace("*", "")

        # Add natural pauses
        tts_result = tts_result.replace(".", ". ")
        tts_result = tts_result.replace(",", ", ")

        # Replace technical terms with pronunciations
        replacements = {
            "°F": " degrees Fahrenheit",
            "°C": " degrees Celsius",
            "mph": " miles per hour",
            "km/h": " kilometers per hour",
            "%": " percent"
        }

        for old, new in replacements.items():
            tts_result = tts_result.replace(old, new)

        # Ensure proper sentence structure
        tts_result = tts_result.strip()
        if not tts_result.endswith('.'):
            tts_result += '.'

        return tts_result

    except Exception as e:
        logger.error(f"TTS formatting failed: {e}")
        return llm_result  # Return original on failure
```

## Implementation Checklist

### Phase 1: Core Infrastructure (Week 1-2)

- [ ] Extend RoleDefinition schema to support parameters and lifecycle configuration
- [ ] Add parameter validation for examples and enum constraints
- [ ] Update RoleRegistry to handle lifecycle functions and parameter schemas
- [ ] Enhance RequestRouter with parameter extraction capabilities
- [ ] Add enhanced routing prompt generation with parameter schemas
- [ ] Implement routing result parsing for parameters
- [ ] Add comprehensive error handling for parameter extraction

### Phase 2: UniversalAgent Enhancement (Week 2-3)

- [ ] Add hybrid execution support to UniversalAgent.execute_task()
- [ ] Implement \_execute_hybrid_task() method with lifecycle phases
- [ ] Add pre-processing execution with parameter injection
- [ ] Add post-processing execution with result transformation
- [ ] Implement data injection for LLM context enhancement
- [ ] Add execution metrics and logging for hybrid roles
- [ ] Add fallback handling for missing lifecycle functions

### Phase 3: Weather Role Migration (Week 3)

- [ ] Convert weather role definition to hybrid pattern
- [ ] Create roles/weather/lifecycle.py with pre/post processors
- [ ] Implement fetch_weather_data pre-processor
- [ ] Implement format_for_tts post-processor
- [ ] Update weather role YAML with parameter schema and lifecycle config
- [ ] Remove weather tools from shared tools (data is pre-fetched)
- [ ] Test weather role parameter extraction and execution

### Phase 4: WorkflowEngine Integration (Week 4)

- [ ] Update WorkflowEngine.\_handle_fast_reply() for hybrid support
- [ ] Add parameter passing from routing to execution
- [ ] Enhance result storage to include extracted parameters
- [ ] Add hybrid role execution metrics to workflow tracking
- [ ] Update complex workflow handling for hybrid roles
- [ ] Add comprehensive error handling and fallbacks

### Phase 5: Testing and Validation (Week 5)

- [ ] Create unit tests for enhanced RequestRouter
- [ ] Create unit tests for UniversalAgent hybrid execution
- [ ] Create integration tests for weather hybrid role
- [ ] Create performance benchmarks for enhanced routing
- [ ] Test parameter extraction accuracy with various inputs
- [ ] Test lifecycle function error handling and recovery
- [ ] Validate end-to-end hybrid role execution

### Phase 6: Documentation and Cleanup (Week 6)

- [ ] Update API documentation for enhanced components
- [ ] Create migration guide for converting roles to hybrid
- [ ] Add examples for common lifecycle patterns
- [ ] Create troubleshooting guide for hybrid roles
- [ ] Update configuration examples and best practices
- [ ] Clean up deprecated code paths
- [ ] Performance optimization based on testing results

## Performance Considerations

### Enhanced Routing Impact

- **Token increase**: ~135% (100 → 235 tokens)
- **Latency increase**: ~135% (2s → 4.7s)
- **Cost increase**: ~135% ($0.0003 → $0.0007 per request)

### Hybrid Execution Benefits

- **Eliminates redundant LLM calls** for data fetching
- **Faster response times** with pre-processed data
- **Better caching opportunities** with structured parameters
- **Reduced overall system complexity**

### Net Performance Impact

The routing overhead is offset by eliminating parameter extraction calls and redundant tool usage in hybrid roles, resulting in overall performance improvement for complex workflows.

## Testing Strategy

### Unit Tests

- **Enhanced RequestRouter parameter extraction**
- **UniversalAgent hybrid lifecycle execution**
- **RoleRegistry lifecycle function management**
- **Parameter validation and enum constraint handling**

### Integration Tests

- **End-to-end weather hybrid role execution**
- **Parameter flow from routing to execution**
