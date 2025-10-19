# Universal Agent System - Roles Directory

This directory contains the specialized roles for the Universal Agent System, implementing a **single-file role architecture** with **Router-Driven Context Selection** capabilities and **lifecycle function support**.

## 🎭 **Role Architecture Overview**

The Universal Agent System uses a **single agent with multiple specialized roles** approach. Each role is implemented as a single Python file following LLM-safe patterns and context-aware design principles.

### **Key Architectural Principles**

- **Single-File Roles**: Each role consolidated into one Python file (~300-700 lines vs 1800+ lines)
- **Context-Aware**: Roles can request and utilize contextual information via Router-Driven Context Selection
- **LLM-Safe**: Designed specifically for AI agent development and modification
- **Intent-Based**: Pure function event handlers returning intents
- **Single Event Loop**: No background threads, no race conditions
- **Lifecycle Support**: Pre/post processing functions for enhanced capabilities

## 🤖 **Available Roles**

### **Core Active Roles**

#### **Timer Role** - [`core_timer.py`](core_timer.py)

- **Purpose**: Timer and alarm management with heartbeat-driven architecture
- **Context Requirements**: None (zero overhead)
- **LLM Type**: WEAK (fast, efficient)
- **Features**: Set timers, cancel timers, list active timers
- **Architecture**: Heartbeat-driven with Redis sorted sets
- **Example**: "Set a timer for 5 minutes" → No context gathering needed

#### **Weather Role** - [`core_weather.py`](core_weather.py)

- **Purpose**: Weather information and forecasts
- **Context Requirements**: Environment context (optional)
- **LLM Type**: DEFAULT
- **Features**: Current weather, forecasts, weather-based recommendations
- **Lifecycle Functions**: Pre-processing (fetch_weather_data), Post-processing (format_for_tts, pii_scrubber)
- **Example**: "What's the weather?" → May use location context if available

#### **Search Role** - [`core_search.py`](core_search.py)

- **Purpose**: Web search and news search via Tavily API
- **Context Requirements**: None
- **LLM Type**: DEFAULT
- **Features**: Web search, news search, search result processing
- **Example**: "Search for latest AI news" → Uses Tavily API for current results

#### **Smart Home Role** - [`core_smart_home.py`](core_smart_home.py)

- **Purpose**: Device control and automation via Home Assistant MCP
- **Context Requirements**: Location context (for room-specific control)
- **LLM Type**: DEFAULT
- **Features**: Device control, automation, status queries
- **Example**: "Turn on the lights" → Router requests location context for current room

#### **Planning Role** - [`core_planning.py`](core_planning.py)

- **Purpose**: Complex task planning and TaskGraph generation
- **Context Requirements**: Memory context (for personalized planning)
- **LLM Type**: STRONG (complex reasoning)
- **Features**: Multi-step planning, task analysis, workflow creation using available system roles
- **Lifecycle Functions**: Pre-processing (load_available_roles), Post-processing (validate_task_graph)
- **Example**: "Plan my morning routine" → Router requests memory context for preferences

#### **Router Role** - [`core_router.py`](core_router.py)

- **Purpose**: Request routing and intelligent context selection
- **Context Requirements**: None (router determines context for other roles)
- **LLM Type**: WEAK (fast routing decisions)
- **Features**: Role selection, context requirement determination, confidence scoring
- **Architecture**: JSON response with Pydantic validation
- **Example**: Analyzes "Turn on lights" → Routes to smart_home + requests location context

#### **Calendar Role** - [`core_calendar.py`](core_calendar.py)

- **Purpose**: Calendar and scheduling management with context awareness
- **Context Requirements**: Schedule context, memory context (optional)
- **LLM Type**: DEFAULT
- **Features**: Schedule retrieval, event creation, calendar queries
- **Context Aware**: Uses memory for recurring events, location for event suggestions
- **Example**: "What's my schedule today?" → Router requests schedule context

#### **Conversation Role** - [`core_conversation.py`](core_conversation.py)

- **Purpose**: Conversation analysis and memory management
- **Context Requirements**: Memory context
- **LLM Type**: DEFAULT
- **Features**: Conversation analysis, topic search, memory storage
- **Example**: "What did we discuss about the project?" → Uses memory context for conversation history

## 🧠 **Router-Driven Context Selection**

The system implements **intelligent context selection** where the router role determines what contextual information is needed for each request.

### **Context Types**

#### **Location Context** (`location`)

- **When Used**: Device control, room-specific actions
- **Data Source**: MQTT location provider (Home Assistant integration)
- **Storage**: Redis with 24-hour TTL
- **Example**: "Turn on the lights" → Router requests location → Uses current room

#### **Memory Context** (`recent_memory`)

- **When Used**: "Usual", "like before", "remember" queries
- **Data Source**: Redis memory provider with LLM-based importance scoring
- **Storage**: Redis with importance-based TTL (30-60 days)
- **Example**: "Play my usual music" → Router requests memory → Recalls music preferences

#### **Presence Context** (`presence`)

- **When Used**: Whole-house actions, privacy considerations
- **Data Source**: MQTT location data for all household members
- **Logic**: Determines who else is currently home
- **Example**: "Turn off all lights" → Router requests presence → Considers other occupants

#### **Schedule Context** (`schedule`)

- **When Used**: Time-sensitive responses, planning queries
- **Data Source**: Redis-cached calendar data
- **Integration**: Calendar role tools for schedule retrieval
- **Example**: "What's my schedule?" → Router requests schedule → Gets today's events

### **Context Selection Logic**

The router uses **surgical context gathering** - only collecting the specific context types needed:

```
User Input → Router Analysis → Context Requirements → Context Gathering → Enhanced Role Execution
```

**Zero Overhead Examples:**

- "Set timer for 5 minutes" → Timer role, no context needed
- "What time is it?" → Planning role, no context needed

**Context-Aware Examples:**

- "Turn on lights" → Smart home role + location context
- "Play my music" → Planning role + memory context
- "Turn off all lights" → Smart home role + location + presence context

## 🏗️ **Role Implementation Pattern**

Each role follows a standardized single-file pattern with 6 main sections:

```python
# roles/core_example.py

"""Example role - LLM-friendly single file implementation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List
from strands import tool
from common.intents import Intent, NotificationIntent
from common.event_context import LLMSafeEventContext

logger = logging.getLogger(__name__)

# 1. ROLE METADATA (replaces definition.yaml)
ROLE_CONFIG = {
    "name": "example",
    "version": "1.0.0",
    "description": "Example role functionality",
    "llm_type": "DEFAULT",
    "fast_reply": True,
    "memory_enabled": False,  # Set to True if role benefits from memory context
    "location_aware": False,  # Set to True if role needs location context
    "presence_aware": False,  # Set to True if role considers household presence
    "schedule_aware": False,  # Set to True if role uses calendar/schedule data
    "when_to_use": "When to use this role",
    "parameters": {
        # Parameter schema for router context selection
        "action": {
            "type": "string",
            "required": True,
            "description": "Action to perform",
            "examples": ["get", "set", "list"]
        }
    },
    "tools": {
        "automatic": True,
        "shared": [],
        "include_builtin": False,
    },
    "prompts": {
        "system": "Role-specific system prompt with clear instructions"
    }
}

# 2. ROLE-SPECIFIC INTENTS
@dataclass
class ExampleIntent(Intent):
    """Role-specific intent."""
    action: str
    data: Dict[str, Any]

    def validate(self) -> bool:
        return bool(self.action)

# 3. EVENT HANDLERS (pure functions returning intents)
def handle_example_event(event_data: Any, context: LLMSafeEventContext) -> List[Intent]:
    """Pure function event handler."""
    return [NotificationIntent(
        message="Event processed",
        channel=context.get_safe_channel()
    )]

# 4. TOOLS
@tool
def example_tool(parameter: str) -> Dict[str, Any]:
    """Example tool function."""
    return {"success": True, "result": parameter}

# 5. LIFECYCLE FUNCTIONS (optional - discovered automatically)
def example_pre_processor(instruction: str, context, parameters: dict) -> dict:
    """Pre-processing function - runs before LLM execution.

    Args:
        instruction: User instruction
        context: Task context
        parameters: Extracted parameters from routing

    Returns:
        dict: Data to inject into LLM prompt
    """
    return {
        "processed_data": f"Pre-processed: {instruction}",
        "timestamp": "2025-01-01T00:00:00Z"
    }

def example_post_processor(llm_result: str, context, pre_data: dict) -> str:
    """Post-processing function - runs after LLM execution.

    Args:
        llm_result: LLM response text
        context: Task context
        pre_data: Data from pre-processing

    Returns:
        str: Final processed result
    """
    return f"Post-processed: {llm_result}"

# 6. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {"EXAMPLE_EVENT": handle_example_event},
        "tools": [example_tool],
        "intents": [ExampleIntent]
    }
```

## 🔧 **Context-Aware Role Development**

### **Making Roles Context-Aware**

To make a role context-aware, set the appropriate flags in `ROLE_CONFIG`:

```python
ROLE_CONFIG = {
    # ... other config ...
    "memory_enabled": True,     # Role benefits from memory context
    "location_aware": True,     # Role needs location context
    "presence_aware": True,     # Role considers household presence
    "schedule_aware": True,     # Role uses calendar/schedule data
}
```

### **Context Usage in Roles**

Context is automatically provided by the system when the router determines it's needed:

```python
# Context is injected into the enhanced prompt
# Original: "Turn on the lights"
# Enhanced: "Turn on the lights\n\nContext: Location: bedroom"

# Roles receive context-enhanced prompts automatically
# No code changes needed in role implementation
```

### **Memory Assessment**

Roles can influence memory storage through interaction importance:

- **High Importance**: Personal information, preferences, commitments
- **Medium Importance**: Device settings, recurring patterns
- **Low Importance**: Simple commands, status queries

The system automatically assesses and stores important interactions using LLM-based scoring.

## 🔄 **Lifecycle Functions**

Lifecycle functions provide pre and post-processing capabilities for roles.

### **Pre-Processing Functions**

Pre-processing functions run **before** LLM execution and can:

- Fetch external data
- Process parameters
- Prepare context for LLM

**Function Signature:**

```python
def function_name(instruction: str, context, parameters: dict) -> dict:
    """Pre-processing function."""
    return {"key": "value"}  # Data injected into LLM prompt
```

**Examples:**

- `fetch_weather_data()` - Fetches weather data before LLM processes request
- `load_available_roles()` - Loads system roles for planning tasks

### **Post-Processing Functions**

Post-processing functions run **after** LLM execution and can:

- Format LLM output
- Scrub sensitive data
- Transform results

**Function Signature:**

```python
def function_name(llm_result: str, context, pre_data: dict) -> str:
    """Post-processing function."""
    return "processed_result"  # Final result returned to user
```

**Examples:**

- `format_for_tts()` - Formats weather responses for text-to-speech
- `pii_scrubber()` - Removes sensitive data from responses
- `validate_task_graph()` - Validates planning output structure

### **Lifecycle Function Discovery**

Lifecycle functions are **automatically discovered** by the Universal Agent:

- **No registration required** - functions are found by naming convention
- **No configuration needed** - not declared in ROLE_CONFIG or register_role()
- **Flexible naming** - any function name works as long as signature matches
- **Optional** - roles work fine without lifecycle functions

## 📊 **Role Performance Characteristics**

### **LLM Type Guidelines**

- **WEAK**: Fast routing, simple operations (Router, Timer)
- **DEFAULT**: Standard role operations (Weather, Smart Home, Calendar, Search, Conversation)
- **STRONG**: Complex reasoning and planning (Planning role)

### **Context Overhead**

- **No Context**: ~0ms overhead (Timer, simple Weather)
- **Location Context**: ~0ms (MQTT push, cached in Redis)
- **Memory Context**: ~15ms (Redis key scan and read)
- **Multiple Contexts**: ~15ms (parallel gathering)

### **Fast Reply Optimization**

Roles marked with `"fast_reply": True` are optimized for quick responses:

- Pre-warmed in agent pool
- Prioritized in router selection
- Minimal context gathering when possible

## 🔍 **Role Discovery and Registration**

The system automatically discovers roles using the `register_role()` function:

1. **Auto-Discovery**: RoleRegistry scans `roles/core_*.py` files
2. **Registration**: Calls `register_role()` function in each file
3. **Validation**: Validates role structure and configuration
4. **Integration**: Registers tools, intents, and event handlers
5. **Lifecycle Discovery**: Finds lifecycle functions by signature matching

### **Role Registry Integration**

```python
# Automatic role discovery pattern
def register_role():
    return {
        "config": ROLE_CONFIG,           # Role metadata and configuration
        "event_handlers": {...},         # Event type → handler function mapping
        "tools": [...],                  # List of @tool decorated functions
        "intents": [...]                 # List of Intent classes
    }
    # Note: Lifecycle functions are NOT declared here - they're auto-discovered
```

## 🧪 **Testing Context-Aware Roles**

### **Unit Testing Pattern**

```python
# Test role configuration
def test_role_config():
    assert ROLE_CONFIG["memory_enabled"] == True
    assert ROLE_CONFIG["location_aware"] == True

# Test context awareness
def test_role_context_integration():
    registration = register_role()
    assert registration["config"]["memory_enabled"] == True

# Test lifecycle functions
def test_pre_processing():
    result = example_pre_processor("test instruction", Mock(), {})
    assert "processed_data" in result

def test_post_processing():
    result = example_post_processor("test result", Mock(), {})
    assert result.startswith("Post-processed:")
```

### **Integration Testing**

Context-aware roles are tested through:

- **Unit Tests**: Role-specific functionality and lifecycle functions
- **Integration Tests**: Context gathering and usage
- **End-to-End Tests**: Complete request flow with context

## 📁 **Directory Structure**

```
roles/
├── README.md                    # This file - role development guide
├── core_timer.py               # Timer role with heartbeat architecture
├── core_weather.py             # Weather role with lifecycle functions
├── core_search.py              # Search role with Tavily API integration
├── core_smart_home.py          # Smart home role with location context
├── core_planning.py            # Planning role with TaskGraph generation
├── core_router.py              # Router role with context selection
├── core_calendar.py            # Calendar role with schedule context
├── core_conversation.py        # Conversation role with memory context
├── shared_tools/               # Shared tool functions
│   ├── __init__.py
│   └── redis_tools.py          # Redis operations for context storage
└── archived/                   # Legacy multi-file roles (deprecated)
    └── ...                     # Archived role implementations
```

## 🚀 **Development Guidelines**

### **For LLM Developers**

1. **Follow Single-File Pattern**: Keep all role logic in one file
2. **Use Context Flags**: Set `memory_enabled`, `location_aware` flags appropriately
3. **Implement Pure Functions**: Event handlers should be pure functions returning intents
4. **Add Lifecycle Functions**: Use pre/post processing for external data and formatting
5. **Type Safety**: Ensure all examples in parameters are properly typed
6. **Test Coverage**: Write comprehensive tests for all role functionality

### **Context Integration**

1. **Router Determines Context**: Don't manually request context in roles
2. **Context is Automatic**: Enhanced prompts include relevant context
3. **Graceful Degradation**: Roles work without context if gathering fails
4. **Memory Assessment**: Important interactions are automatically stored

### **Lifecycle Function Guidelines**

1. **Optional Enhancement**: Lifecycle functions are optional - roles work without them
2. **Automatic Discovery**: No need to register or configure lifecycle functions
3. **Proper Signatures**: Follow exact function signatures for pre/post processing
4. **Error Handling**: Include proper error handling in lifecycle functions
5. **Sync by Default**: Lifecycle functions should be synchronous unless specifically needed async

### **Performance Considerations**

1. **Fast Reply**: Mark simple roles with `"fast_reply": True`
2. **LLM Type**: Use appropriate LLM strength for role complexity
3. **Context Overhead**: Consider context gathering cost in role design
4. **Tool Selection**: Use `"automatic": True` for role-specific tools only
5. **Lifecycle Efficiency**: Keep lifecycle functions fast and focused

## 📖 **Related Documentation**

- **[Document 33](../docs/33_ROUTER_DRIVEN_CONTEXT_SELECTION_DESIGN.md)**: Complete Router-Driven Context Selection design
- **[Document 34](../docs/34_PLANNING_ROLE_DESIGN.md)**: Planning role implementation with lifecycle functions
- **[Architecture Overview](../docs/01_ARCHITECTURE_OVERVIEW.md)**: System architecture patterns
- **[Tool Development Guide](../docs/05_TOOL_DEVELOPMENT_GUIDE.md)**: Creating new tools and roles

The roles directory implements a sophisticated context-aware agent system that provides intelligent, personalized responses while maintaining LLM-safe architecture principles and zero-overhead performance for simple requests. The addition of lifecycle functions enables powerful pre and post-processing capabilities while maintaining the simplicity of the single-file role architecture.
