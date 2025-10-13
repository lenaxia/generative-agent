# Role Creation Guide for Universal Agent System

This guide provides complete instructions for creating new roles in the Universal Agent System. It is designed to be comprehensive enough that reading this document alone provides all necessary information to implement a new role.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Role Implementation Patterns](#role-implementation-patterns)
3. [Single-File Role Structure](#single-file-role-structure)
4. [Step-by-Step Role Creation](#step-by-step-role-creation)
5. [Pattern-Specific Implementation](#pattern-specific-implementation)
6. [Testing and Validation](#testing-and-validation)
7. [Communication Architecture](#communication-architecture)
8. [Common Patterns and Examples](#common-patterns-and-examples)

## Architecture Overview

### Core Principles

The Universal Agent System uses a **single-file role architecture** with these principles:

- **LLM-Safe Design**: Each role is completely self-contained in one Python file
- **Intent-Based Processing**: Pure function event handlers return declarative intents
- **Single Event Loop**: No threading complexity or race conditions
- **Tool Configuration Control**: Each role specifies its own tool requirements

### Role Lifecycle

1. **Auto-Discovery**: RoleRegistry automatically discovers roles via `register_role()` function
2. **Configuration**: Role metadata defines capabilities, tools, and LLM requirements
3. **Execution**: UniversalAgent assembles tools and executes role-specific logic
4. **Intent Processing**: Role returns intents that are processed by infrastructure

## Role Implementation Patterns

There are three distinct patterns for implementing roles:

## ⚠️ **CRITICAL: Built-in Tools Should Be Excluded**

**IMPORTANT**: Most roles should set `"include_builtin": False` to exclude calculator, file_read, and shell tools.

### Why Exclude Built-in Tools?

- **Role Specialization**: Each role should have a focused, specific purpose
- **LLM Confusion**: Extra tools can confuse the LLM about the role's intended function
- **Security**: Roles shouldn't have unnecessary system access (shell, file operations)
- **Performance**: Fewer tools mean faster decision-making and cleaner responses

### Built-in Tools Available

The system provides these built-in tools that are **usually excluded**:

- **`calculator`**: Basic mathematical operations
- **`file_read`**: File system read operations
- **`shell`**: System shell command execution

### When to Include Built-in Tools

**Rarely needed** - only include if your role specifically requires:

- Mathematical calculations (`calculator`)
- File system access (`file_read`)
- System operations (`shell`)

**Default recommendation**: `"include_builtin": False` for all new roles.

---

### Pattern 1: JSON Response (Classification/Routing)

**Use for**: Decision-making, classification, routing, analysis tasks
**Tools**: None - LLM outputs structured JSON
**Flow**: Request → LLM → JSON Response → Pydantic Validation → Result

```python
# Example: Router role - EXCLUDES built-in tools
"tools": {
    "automatic": False,     # No custom tools
    "shared": [],          # No shared tools
    "include_builtin": False,  # CRITICAL: Exclude calculator, file_read, shell
}
```

### Pattern 2: Pre-Processing (External Data Integration)

**Use for**: Roles requiring external data (APIs, databases, services)
**Tools**: None - data pre-fetched and injected into prompts
**Flow**: Request → Pre-fetch Data → Data Injection → LLM → Post-process → Result

```python
# Example: Weather role - EXCLUDES built-in tools
"tools": {
    "automatic": False,     # No custom tools
    "shared": [],          # No shared tools
    "include_builtin": False,  # CRITICAL: Exclude calculator, file_read, shell
}
```

### Pattern 3: Tool-Based (Action-Oriented)

**Use for**: Roles that need to perform operations or actions
**Tools**: Custom tools with `@tool` decorator
**Flow**: Request → LLM → Tool Calls → Tool Execution → Results

```python
# Example: Timer role - EXCLUDES built-in tools (typical)
"tools": {
    "automatic": True,      # Include custom tools
    "shared": ["redis_tools"],  # Include specific shared tools
    "include_builtin": False,   # CRITICAL: Usually exclude calculator, file_read, shell
}
```

## Single-File Role Structure

Every role must follow this exact 10-section structure:

```python
"""Role Name - LLM-friendly single file implementation.

Brief description of role functionality and purpose.

Architecture: Single Event Loop + Intent-Based + [Pattern Type]
Created: [Date]
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Pattern-specific imports
from pydantic import BaseModel, Field, ValidationError  # For JSON Response pattern
from strands import tool  # For Tool-Based pattern
import requests  # For Pre-Processing pattern

from common.enhanced_event_context import LLMSafeEventContext
from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. PYDANTIC MODELS (JSON Response pattern only)
# 2. ROLE METADATA
# 3. ROLE-SPECIFIC INTENTS
# 4. EVENT HANDLERS
# 5. PATTERN-SPECIFIC FUNCTIONS
# 6. HELPER FUNCTIONS
# 7. INTENT HANDLER REGISTRATION
# 8. ROLE REGISTRATION
# 9. CONSTANTS AND CONFIGURATION
# 10. ERROR HANDLING UTILITIES
```

## Step-by-Step Role Creation

### Step 1: Choose Implementation Pattern

Determine which pattern fits your role:

- **JSON Response**: For classification, routing, decision-making
- **Pre-Processing**: For external data integration (APIs, databases)
- **Tool-Based**: For action-oriented operations

### Step 2: Create Role File

Create `roles/your_role_name_single_file.py` with the 10-section structure.

### Step 3: Implement Required Sections

#### Section 1: Pydantic Models (JSON Response only)

```python
# 1. PYDANTIC MODELS (JSON Response pattern only)
class YourRoleResponse(BaseModel):
    """Pydantic model for parsing LLM responses."""

    field_name: str = Field(..., description="Field description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

    class Config:
        extra = "forbid"  # Don't allow extra fields
```

#### Section 2: Role Metadata (Required)

```python
# 2. ROLE METADATA (replaces definition.yaml)
ROLE_CONFIG = {
    "name": "your_role_name",  # Must match filename without _single_file.py
    "version": "1.0.0",
    "description": "Clear description of role capabilities and purpose",
    "llm_type": "WEAK|DEFAULT|STRONG",  # Choose based on complexity
    "fast_reply": True|False,  # True for quick responses
    "when_to_use": "Specific criteria for when this role should be selected",
    "tools": {
        "automatic": True|False,  # Include custom @tool functions?
        "shared": ["tool_name"],  # List of shared tools to include
        "include_builtin": True|False,  # Include calculator, file_read, shell?
    },
    "prompts": {
        "system": """System prompt for the role.

        For JSON roles: Include Backus-Naur form specification.
        For tool roles: Describe available tools and usage.
        For pre-processing roles: Explain that data is pre-fetched."""
    },
}
```

#### Section 3: Role-Specific Intents (Required)

```python
# 3. ROLE-SPECIFIC INTENTS (owned by this role)
@dataclass
class YourRoleIntent(Intent):
    """Role-specific intent - owned by this role."""

    action: str  # "action1", "action2", etc.
    parameters: Dict[str, Any]

    def validate(self) -> bool:
        """Validate intent parameters."""
        return bool(self.action and isinstance(self.parameters, dict))
```

#### Section 4: Event Handlers (Required)

```python
# 4. EVENT HANDLERS (pure functions returning intents)
def handle_role_event(event_data: Any, context: LLMSafeEventContext) -> List[Intent]:
    """LLM-SAFE: Pure function for role-specific events."""
    try:
        # Parse event data
        parsed_data = _parse_event_data(event_data)

        # Create intents (no side effects)
        return [
            NotificationIntent(
                message=f"Event processed: {parsed_data}",
                channel=context.get_safe_channel(),
                user_id=context.user_id,
                priority="medium",
            ),
            AuditIntent(
                action="event_handled",
                details={"data": parsed_data, "processed_at": time.time()},
                user_id=context.user_id,
                severity="info",
            ),
        ]

    except Exception as e:
        logger.error(f"Role event handler error: {e}")
        return [
            NotificationIntent(
                message=f"Event processing error: {e}",
                channel=context.get_safe_channel(),
                priority="high",
                notification_type="error",
            )
        ]
```

#### Section 5: Pattern-Specific Functions (Required)

This section varies by pattern - see [Pattern-Specific Implementation](#pattern-specific-implementation).

#### Section 6: Helper Functions (Optional)

```python
# 6. HELPER FUNCTIONS (minimal, focused)
def _parse_event_data(event_data: Any) -> Dict[str, Any]:
    """LLM-SAFE: Parse event data with error handling."""
    try:
        if isinstance(event_data, dict):
            return event_data
        elif isinstance(event_data, str):
            return {"text": event_data}
        else:
            return {"raw": str(event_data)}
    except Exception as e:
        return {"error": str(e)}
```

#### Section 7: Intent Handler Registration (Optional)

```python
# 7. INTENT HANDLER REGISTRATION (if using custom intents)
async def process_role_intent(intent: YourRoleIntent):
    """Process role-specific intents - called by IntentProcessor."""
    logger.info(f"Processing role intent: {intent.action}")

    # Implementation depends on intent action
    # This is called by the infrastructure when intents are processed
```

#### Section 8: Role Registration (Required)

```python
# 8. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "ROLE_EVENT": handle_role_event,
            # Add other event handlers as needed
        },
        "tools": [role_tool] if ROLE_CONFIG["tools"]["automatic"] else [],
        "intents": {
            YourRoleIntent: process_role_intent,
            # Add other intent handlers as needed
        },
    }
```

#### Section 9: Constants and Configuration (Optional)

```python
# 9. CONSTANTS AND CONFIGURATION
ROLE_CONSTANTS = {
    "timeout_seconds": 30,
    "max_retries": 3,
    "default_priority": "medium",
}
```

#### Section 10: Error Handling Utilities (Optional)

```python
# 10. ERROR HANDLING UTILITIES
def _create_error_response(error: Exception, context: str = "") -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "success": False,
        "error": str(error),
        "error_type": error.__class__.__name__,
        "context": context,
        "timestamp": time.time(),
    }
```

## Pattern-Specific Implementation

### JSON Response Pattern Implementation

For roles that need to output structured data (routing, classification, analysis):

#### Section 5: JSON Response Functions

```python
# 5. JSON RESPONSE FUNCTIONS
def parse_role_response(response_text: str) -> Dict[str, Any]:
    """Parse role response using Pydantic validation."""
    try:
        role_response = YourRoleResponse.model_validate_json(response_text)
        return {
            "valid": True,
            "data": role_response.dict(),
        }
    except ValidationError as e:
        logger.error(f"Response parsing failed: {e}")
        return {
            "valid": False,
            "error": f"Validation error: {str(e)}",
            "fallback_data": {}
        }

def process_request_with_json_response(request_text: str, parameters: Dict[str, Any]) -> str:
    """Process request and return JSON response."""
    try:
        # Build instruction for JSON-only response
        instruction = f"""USER REQUEST: "{request_text}"

        Respond with ONLY valid JSON in this exact format:
        {{
          "field_name": "value",
          "confidence": 0.95
        }}"""

        # Execute with role
        from llm_provider.factory import LLMType
        universal_agent = getattr(process_request_with_json_response, "_universal_agent", None)

        if not universal_agent:
            return '{"error": "No universal agent available"}'

        result = universal_agent.execute_task(
            instruction=instruction,
            role="your_role_name",
            llm_type=LLMType.WEAK
        )

        # Parse and validate JSON response
        parsed = parse_role_response(result)
        if parsed["valid"]:
            return result
        else:
            return f'{{"error": "{parsed["error"]}"}}'

    except Exception as e:
        logger.error(f"JSON response processing failed: {e}")
        return f'{{"error": "{str(e)}"}}'
```

#### System Prompt for JSON Response

```python
"prompts": {
    "system": """You are a specialized agent that responds with ONLY valid JSON.

CRITICAL: Respond with ONLY valid JSON. No explanations, no additional text.

Response format (Backus-Naur form):
<response> ::= "{" <field1> "," <field2> "}"
<field1> ::= '"field_name":' <string_value>
<field2> ::= '"confidence":' <number_between_0_and_1>

Example:
{
  "field_name": "example_value",
  "confidence": 0.95
}

Always validate your JSON before responding."""
}
```

### Pre-Processing Pattern Implementation

For roles that need external data (APIs, databases, services):

#### Section 5: Pre-Processing Functions

```python
# 5. PRE-PROCESSING FUNCTIONS
def fetch_external_data(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-processing: Fetch external data for injection into prompt."""
    try:
        # Extract parameters from routing decision
        location = parameters.get("location", "")
        data_type = parameters.get("type", "default")

        # Fetch data from external APIs, databases, etc.
        api_url = f"https://api.example.com/data?location={location}&type={data_type}"
        headers = {"User-Agent": "UniversalAgent/1.0"}

        response = requests.get(api_url, headers=headers, timeout=10)
        external_data = response.json()

        return {
            "success": True,
            "data": external_data,
            "timestamp": time.time(),
            "source": "external_api"
        }
    except Exception as e:
        logger.error(f"Pre-processing data fetch failed: {e}")
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }

def process_request_with_injected_data(request_text: str, parameters: Dict[str, Any]) -> str:
    """Pre-processing: Process request with pre-fetched data injection."""
    try:
        # 1. Pre-processing: Fetch external data
        data_result = fetch_external_data(parameters)

        if not data_result["success"]:
            return f"Unable to process request: {data_result['error']}"

        # 2. Data injection: Build enhanced prompt with fetched data
        injected_data = data_result["data"]
        enhanced_instruction = f"""USER REQUEST: "{request_text}"

EXTERNAL DATA:
{format_data_for_injection(injected_data)}

Based on this data, provide a comprehensive and helpful response."""

        # 3. LLM processing: Execute with injected data
        from llm_provider.factory import LLMType
        universal_agent = getattr(process_request_with_injected_data, '_universal_agent', None)

        if not universal_agent:
            return "Data fetched but processing unavailable"

        result = universal_agent.execute_task(
            instruction=enhanced_instruction,
            role="your_role_name",
            llm_type=LLMType.WEAK
        )

        # 4. Post-processing: Format result if needed
        return post_process_result(result, parameters)

    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        return f"Processing error: {str(e)}"

def post_process_result(llm_result: str, parameters: Dict[str, Any]) -> str:
    """Post-processing: Format and enhance LLM result."""
    try:
        # Apply role-specific formatting
        formatted_result = apply_role_formatting(llm_result)

        # Add metadata or additional context if needed
        if parameters.get("include_metadata", False):
            formatted_result += f"\n\nProcessed at: {datetime.now().isoformat()}"

        return formatted_result

    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        return llm_result  # Return original result if post-processing fails

def format_data_for_injection(data: Any) -> str:
    """Helper: Format external data for prompt injection."""
    if isinstance(data, dict):
        return "\n".join([f"- {key}: {value}" for key, value in data.items()])
    elif isinstance(data, list):
        return "\n".join([f"- {item}" for item in data])
    else:
        return str(data)

def apply_role_formatting(result: str) -> str:
    """Apply role-specific formatting to LLM result."""
    # Implement role-specific formatting logic
    return result.strip()
```

#### System Prompt for Pre-Processing

```python
"prompts": {
    "system": """You are a specialized agent with access to pre-fetched external data.

The user request will be accompanied by relevant external data that has been
pre-fetched for you. Use this data to provide accurate, comprehensive responses.

EXTERNAL DATA FORMAT:
The data will be provided in the prompt in a structured format. Always reference
this data in your response and ensure accuracy based on the provided information.

Your responses should be:
- Based on the provided external data
- Comprehensive and helpful
- Formatted appropriately for the user
- Include relevant details from the external data"""
}
```

### Tool-Based Pattern Implementation

For roles that need to perform actions or operations:

#### Section 5: Tool Functions

```python
# 5. TOOL FUNCTIONS
@tool
def role_action_tool(parameter: str, option: str = "default") -> Dict[str, Any]:
    """LLM-SAFE: Role-specific action tool."""
    try:
        # Validate parameters
        if not parameter or not parameter.strip():
            return {
                "success": False,
                "error": "Parameter cannot be empty",
                "timestamp": time.time()
            }

        # Perform the action
        result = perform_role_action(parameter, option)

        return {
            "success": True,
            "result": result,
            "parameter": parameter,
            "option": option,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }

@tool
def role_query_tool(query: str) -> Dict[str, Any]:
    """LLM-SAFE: Role-specific query tool."""
    try:
        # Process query
        query_result = process_role_query(query)

        return {
            "success": True,
            "query": query,
            "result": query_result,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Query tool failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }

def perform_role_action(parameter: str, option: str) -> Any:
    """Implement the actual role action logic."""
    # This is where you implement the core functionality
    # Examples: set timer, control device, send message, etc.
    pass

def process_role_query(query: str) -> Any:
    """Implement the actual query processing logic."""
    # This is where you implement query handling
    # Examples: search data, retrieve status, check conditions, etc.
    pass
```

#### System Prompt for Tool-Based

```python
"prompts": {
    "system": """You are a specialized agent with access to specific tools for performing actions.

AVAILABLE TOOLS:
- role_action_tool(parameter, option): Performs the main role action
- role_query_tool(query): Queries information related to the role

TOOL USAGE:
- Always use tools to perform actions rather than just describing them
- Validate tool results and handle errors appropriately
- Provide clear feedback about tool execution results
- Use multiple tools in sequence if needed for complex tasks

Your responses should:
- Use tools to accomplish the requested actions
- Provide clear status updates about tool execution
- Handle errors gracefully with helpful error messages
- Confirm successful completion of actions"""
}
```

## Communication Architecture

### Channel Context Preservation

**CRITICAL**: Roles that send notifications must preserve the original channel context to ensure messages are delivered back to the correct channel (e.g., Slack, console, etc.).

#### The Problem

When a role processes a request from a specific channel (e.g., Slack `#general`), any notifications or responses must be routed back to that same channel. However, the communication system uses a **separation of concerns** architecture where:

- **CommunicationManager**: Routes messages to channels (no channel-specific logic)
- **Channel Handlers**: Handle channel-specific logic and patterns
- **Roles**: Store and preserve original channel context

#### The Solution: Proper Channel Context Flow

```python
# 1. CONTEXT PRESERVATION IN ROLE DATA
# When storing data (timers, tasks, etc.), preserve full channel context
role_data = {
    "id": item_id,
    "user_id": context.user_id,           # e.g., "U12345" (Slack user)
    "channel": context.channel_id,        # e.g., "slack:#general" (full channel ID)
    "created_at": time.time(),
    # ... other role-specific data
}

# 2. CONTEXT RETRIEVAL FOR NOTIFICATIONS
# When sending notifications, use stored channel context (NOT hardcoded defaults)
def send_role_notification(stored_data: dict):
    user_id = stored_data.get("user_id", "system")      # Use stored user
    channel = stored_data.get("channel", "console")     # Use stored channel (NOT hardcoded)

    # Send via message bus with preserved context
    message_payload = {
        "message": "Your notification message",
        "context": {
            "channel_id": channel,        # e.g., "slack:#general"
            "user_id": user_id,          # e.g., "U12345"
            "request_id": f"notification_{item_id}"
        }
    }
```

#### Communication Manager Architecture

The CommunicationManager uses **pattern-based routing** to maintain separation of concerns:

```python
# CommunicationManager routes to full channel IDs (no extraction)
def _determine_target_channels(self, origin_channel: str, message_type: str, context: dict):
    # Route back to origin channel (let channel handlers decide)
    return [origin_channel] if origin_channel else ["console"]

# Channel handlers own their routing patterns
def _find_handler_for_channel(self, channel_id: str) -> Optional[ChannelHandler]:
    """Find appropriate handler based on channel patterns."""
    for handler in self.channels.values():
        if hasattr(handler, 'channel_pattern'):
            if channel_id.startswith(handler.channel_pattern):
                return handler
    return self.channels.get('console')  # Fallback
```

#### Channel Handler Patterns

Each channel handler declares its pattern during registration:

```python
# In SlackHandler
class SlackHandler(ChannelHandler):
    def __init__(self):
        self.channel_pattern = "slack:"  # Handles all "slack:*" channels

# In ConsoleHandler
class ConsoleHandler(ChannelHandler):
    def __init__(self):
        self.channel_pattern = "console"  # Handles "console" channel
```

#### Role Implementation Requirements

**For Timer-like Roles** (storing data for later notification):

```python
# ✅ CORRECT: Store full channel context
timer_data = {
    "user_id": context.user_id,           # From original request context
    "channel": context.channel_id,        # Full channel ID: "slack:#general"
}

# ✅ CORRECT: Use stored context for notifications
def handle_timer_expiry(timer_data: dict):
    channel = timer_data.get("channel", "console")  # Use stored channel
    user_id = timer_data.get("user_id", "system")   # Use stored user

    # Notification will route back to original Slack channel
```

**For Immediate Response Roles** (weather, routing, etc.):

```python
# ✅ CORRECT: Context automatically preserved by workflow engine
# No special handling needed - responses automatically route back to origin
```

#### Common Mistakes to Avoid

```python
# ❌ WRONG: Hardcoded channel defaults
channel = "console"  # This loses original Slack context

# ❌ WRONG: Channel type extraction in CommunicationManager
channel_type = origin_channel.split(":", 1)[0]  # Violates separation of concerns

# ❌ WRONG: Missing context preservation
# Not storing user_id and channel_id when creating timers/tasks
```

#### Testing Channel Context

Verify your role preserves context correctly:

```python
def test_channel_context_preservation():
    """Test that role preserves original channel context."""
    # Simulate Slack request context
    context = MockContext(
        user_id="U12345",
        channel_id="slack:#general"
    )

    # Create role item (timer, task, etc.)
    result = your_role_function("test", context)

    # Verify context is stored
    stored_data = get_stored_data(result["id"])
    assert stored_data["user_id"] == "U12345"
    assert stored_data["channel"] == "slack:#general"

    # Verify notification uses stored context
    notification = simulate_notification(stored_data)
    assert notification["context"]["channel_id"] == "slack:#general"
    assert notification["context"]["user_id"] == "U12345"
```

This architecture ensures notifications are delivered to the correct channels while maintaining proper separation of concerns between the CommunicationManager and channel handlers.

## Testing and Validation

### Basic Role Testing

Create a simple test to verify your role works:

```python
# test_your_role.py
import pytest
from roles.your_role_single_file import register_role, ROLE_CONFIG

def test_role_registration():
    """Test that role registers correctly."""
    registration = register_role()

    assert "config" in registration
    assert "event_handlers" in registration
    assert "tools" in registration
    assert "intents" in registration

    config = registration["config"]
    assert config["name"] == "your_role_name"
    assert config["version"] is not None
    assert config["description"] is not None

def test_role_config_validation():
    """Test that role configuration is valid."""
    assert ROLE_CONFIG["name"] is not None
    assert ROLE_CONFIG["llm_type"] in ["WEAK", "DEFAULT", "STRONG"]
    assert isinstance(ROLE_CONFIG["fast_reply"], bool)
    assert "tools" in ROLE_CONFIG

    tools_config = ROLE_CONFIG["tools"]
    assert isinstance(tools_config["automatic"], bool)
    assert isinstance(tools_config["shared"], list)
    assert isinstance(tools_config["include_builtin"], bool)

def test_intent_validation():
    """Test that role-specific intents validate correctly."""
    from roles.your_role_single_file import YourRoleIntent

    # Valid intent
    valid_intent = YourRoleIntent(
        action="test_action",
        parameters={"key": "value"}
    )
    assert valid_intent.validate()

    # Invalid intent
    invalid_intent = YourRoleIntent(
        action="",
        parameters={}
    )
    assert not invalid_intent.validate()
```

### Integration Testing

Test your role with the Universal Agent:

```python
def test_role_integration():
    """Test role integration with Universal Agent."""
    from llm_provider.role_registry import RoleRegistry
    from llm_provider.universal_agent import UniversalAgent

    # Initialize components
    role_registry = RoleRegistry()

    # Verify role is discovered
    role = role_registry.get_role("your_role_name")
    assert role is not None
    assert role.name == "your_role_name"

    # Test tool assembly (if applicable)
    if ROLE_CONFIG["tools"]["automatic"]:
        assert len(role.custom_tools) > 0
```

## Common Patterns and Examples

### Error Handling Pattern

```python
def safe_operation(operation_name: str, operation_func, *args, **kwargs) -> Dict[str, Any]:
    """Standard error handling pattern for role operations."""
    try:
        logger.info(f"Starting {operation_name}")
        result = operation_func(*args, **kwargs)
        logger.info(f"Completed {operation_name} successfully")
        return {
            "success": True,
            "result": result,
            "operation": operation_name,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed {operation_name}: {e}")
        return {
            "success": False,
            "error": str(e),
            "operation": operation_name,
            "timestamp": time.time()
        }
```

### Configuration Validation Pattern

```python
def validate_role_config(config: Dict[str, Any]) -> List[str]:
    """Validate role configuration and return list of errors."""
    errors = []

    required_fields = ["name", "version", "description", "llm_type", "fast_reply", "tools"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    if "llm_type" in config and config["llm_type"] not in ["WEAK", "DEFAULT", "STRONG"]:
        errors.append(f"Invalid llm_type: {config['llm_type']}")

    if "tools" in config:
        tools_config = config["tools"]
        required_tool_fields = ["automatic", "shared", "include_builtin"]
        for field in required_tool_fields:
            if field not in tools_config:
                errors.append(f"Missing tools field: {field}")

    return errors
```

### Intent Creation Pattern

```python
def create_success_intents(action: str, result: Any, context: LLMSafeEventContext) -> List[Intent]:
    """Standard pattern for creating success intents."""
    return [
        NotificationIntent(
            message=f"Successfully completed {action}",
            channel=context.get_safe_channel(),
            user_id=context.user_id,
            priority="medium",
            notification_type="success"
        ),
        AuditIntent(
            action=f"{action}_completed",
            details={"result": str(result), "timestamp": time.time()},
            user_id=context.user_id,
            severity="info"
        )
    ]

def create_error_intents(action: str, error: Exception, context: LLMSafeEventContext) -> List[Intent]:
    """Standard pattern for creating error intents."""
    return [
        NotificationIntent(
            message=f"Failed to complete {action}: {str(error)}",
            channel=context.get_safe_channel(),
            user_id=context.user_id,
            priority="high",
            notification_type="error"
        ),
        AuditIntent(
            action=f"{action}_failed",
            details={"error": str(error), "error_type": error.__class__.__name__},
            user_id=context.user_id,
            severity="error"
        )
    ]
```

## Role Creation Checklist

When creating a new role, ensure you have:

- [ ] Chosen the appropriate implementation pattern (JSON/Pre-Processing/Tool-Based)
- [ ] Created the single file with all 10 required sections
- [ ] Implemented `ROLE_CONFIG` with all required fields
- [ ] **CRITICAL: Set `"include_builtin": False` to exclude calculator, file_read, shell tools**
- [ ] Created role-specific intents with validation
- [ ] Implemented event handlers as pure functions
- [ ] Added pattern-specific functions (JSON parsing, pre-processing, or tools)
- [ ] Implemented the `register_role()` function correctly
- [ ] Added appropriate error handling throughout
- [ ] Created basic tests for the role
- [ ] Validated the role integrates with the Universal Agent
- [ ] Documented the role's purpose and usage clearly
- [ ] **Verified that built-in tools are excluded unless specifically needed**

Following this guide ensures your role will integrate seamlessly with the Universal Agent System and follow all architectural best practices.
