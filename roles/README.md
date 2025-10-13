# Roles Directory - LLM-Safe Single-File Role Architecture

This directory contains AI agent roles following the **LLM-safe single-file architecture** patterns from Documents 25 & 26. Each role is completely self-contained in a single Python file for maximum LLM-friendliness and maintainability.

## 🏗️ **Architecture Principles**

### **Single-File Role Pattern**

- **One file per role**: All functionality consolidated into a single Python file
- **LLM-friendly**: AI agents can understand and modify entire roles easily
- **Self-contained**: Each role owns all its concerns and dependencies
- **Consistent structure**: Same 10-section pattern for every role

### **Intent-Based Processing**

- **Pure function event handlers**: Return intents, no side effects
- **Declarative intents**: Describe "what should happen", not "how to do it"
- **Infrastructure processing**: Intent processor handles the "how to do it" part

### **Tool Configuration Control**

- **Role-controlled**: Each role specifies its own tool requirements
- **No hardcoded lists**: UniversalAgent respects role tool configuration
- **Flexible options**: Roles can include/exclude built-in tools as needed

## 📁 **Current Roles**

### **Single-File Roles (New Architecture)**

- **[`router_single_file.py`](router_single_file.py)** - Request routing with JSON response and Pydantic parsing
- **[`timer_single_file.py`](timer_single_file.py)** - Timer and alarm management with event-driven workflows
- **[`weather_single_file.py`](weather_single_file.py)** - Weather information with pre-processing data injection
- **[`smart_home_single_file.py`](smart_home_single_file.py)** - Smart home device control and automation
- **[`planning_single_file.py`](planning_single_file.py)** - Complex task planning and analysis

### **Multi-File Roles (Legacy)**

- **`calendar/`** - Event management and scheduling
- **`code_reviewer/`** - Code quality assessment and security analysis
- **`coding/`** - Code generation, debugging, and software development
- **`research_analyst/`** - Comprehensive research and evidence-based reports
- **`search/`** - Web search and information retrieval

## 🛠️ **Creating New Roles**

### **Single-File Role Template**

Use this template for creating new LLM-safe roles:

```python
"""Role Name - LLM-friendly single file implementation.

This role consolidates all [role] functionality into a single file following
the LLM-safe architecture patterns from Documents 25, 26, and 27.

Architecture: Single Event Loop + Intent-Based + [Tool Strategy]
Created: [Date]
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError  # If using structured parsing
from strands import tool  # If using tools

from common.enhanced_event_context import LLMSafeEventContext
from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. PYDANTIC MODELS (if using structured parsing)
class RoleResponse(BaseModel):
    """Pydantic model for parsing LLM responses."""

    field_name: str = Field(..., description="Field description")
    # Add other fields as needed

# 2. ROLE METADATA (replaces definition.yaml)
ROLE_CONFIG = {
    "name": "role_name",
    "version": "1.0.0",
    "description": "Role description and capabilities",
    "llm_type": "WEAK|DEFAULT|STRONG",  # Choose based on complexity
    "fast_reply": True|False,  # True for fast-reply roles
    "when_to_use": "When to use this role - clear criteria",
    "tools": {
        "automatic": True|False,  # Include custom tools?
        "shared": [],  # List of shared tool names to include
        "include_builtin": True|False,  # Include calculator, file_read, shell?
    },
    "prompts": {
        "system": """System prompt for the role.

        For JSON-only roles: Include Backus-Naur form specification.
        For tool-based roles: Describe available tools and usage.
        For pre-processing roles: Explain that data is pre-fetched."""
    },
}

# 3. ROLE-SPECIFIC INTENTS (owned by this role)
@dataclass
class RoleSpecificIntent(Intent):
    """Role-specific intent - owned by this role."""

    action: str  # "action1", "action2", etc.
    parameters: Dict[str, Any]

    def validate(self) -> bool:
        """Validate intent parameters."""
        return bool(self.action and isinstance(self.parameters, dict))

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

# 5. TOOLS (if using tools) OR FUNCTIONS (if using pre-processing)
@tool  # Only if role uses tools
def role_tool(parameter: str) -> Dict[str, Any]:
    """LLM-SAFE: Role-specific tool."""
    try:
        # Tool implementation
        return {
            "success": True,
            "message": f"Action completed: {parameter}",
            "data": {"result": parameter}
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# OR for pre-processing roles:
def fetch_external_data(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch external data for injection into prompt."""
    try:
        # Fetch data from APIs, databases, etc.
        return {"data": "fetched_data", "status": "success"}
    except Exception as e:
        logger.error(f"Data fetching failed: {e}")
        return {"data": None, "status": "error", "error": str(e)}

# 6. PARSING FUNCTIONS (if using structured responses)
def parse_role_response(response_text: str) -> Dict[str, Any]:
    """Parse role response using Pydantic validation."""
    try:
        role_response = RoleResponse.model_validate_json(response_text)
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

# 7. HELPER FUNCTIONS (minimal, focused)
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

# 8. INTENT HANDLER REGISTRATION (if using custom intents)
async def process_role_intent(intent: RoleSpecificIntent):
    """Process role-specific intents - called by IntentProcessor."""
    logger.info(f"Processing role intent: {intent.action}")

    # In full implementation, this would:
    # - Execute the intent action
    # - Update role-specific state
    # - Return additional intents if needed

# 9. ROLE REGISTRATION (auto-discovery)
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
            RoleSpecificIntent: process_role_intent,
            # Add other intent handlers as needed
        },
    }

# 10. CONSTANTS AND CONFIGURATION
ROLE_CONSTANTS = {
    "timeout_seconds": 30,
    "max_retries": 3,
    "default_priority": "medium",
}

# Additional role-specific constants as needed
```

## 🎯 **Role Implementation Strategies**

### **Strategy 1: JSON Response (Router Pattern)**

**Use for**: Routing, classification, decision-making roles
**Tools**: None (JSON response only)
**Flow**: LLM → JSON → Pydantic parsing → Structured result

```python
"tools": {
    "automatic": False,
    "shared": [],
    "include_builtin": False,  # No built-in tools
},
"prompts": {
    "system": """Respond with ONLY valid JSON.

    <response> ::= "{" <field1> "," <field2> "}"
    # Backus-Naur form specification
    """
}
```

### **Strategy 2: Pre-Processing (Weather Pattern)**

**Use for**: External data integration, API-dependent roles
**Tools**: None (data pre-fetched and injected)
**Flow**: Fetch data → Inject into prompt → LLM processes → Direct response

```python
"tools": {
    "automatic": False,
    "shared": [],
    "include_builtin": False,  # No built-in tools
},

def fetch_external_data(parameters):
    # Fetch from APIs, databases, etc.
    return weather_data

def process_with_injected_data(request, data):
    # Inject data into prompt and execute LLM
    return llm_response
```

### **Strategy 3: Tool-Based (Timer Pattern)**

**Use for**: Action-oriented roles that need to perform operations
**Tools**: Custom tools with @tool decorator
**Flow**: LLM → Tool calls → Tool execution → Results

```python
"tools": {
    "automatic": True,  # Include custom tools
    "shared": ["shared_tool_name"],  # Include shared tools
    "include_builtin": False,  # Usually exclude built-in tools
},

@tool
def role_action(parameter: str) -> Dict[str, Any]:
    # Tool implementation
    return {"success": True, "result": result}
```

## 🔧 **Tool Configuration Options**

### **Built-in Tools Control**

```python
"tools": {
    "include_builtin": True,   # Include calculator, file_read, shell
    "include_builtin": False,  # Exclude built-in tools (specialized roles)
}
```

### **Custom Tools Control**

```python
"tools": {
    "automatic": True,   # Include all @tool decorated functions
    "automatic": False,  # No custom tools (JSON/pre-processing roles)
}
```

### **Shared Tools Control**

```python
"tools": {
    "shared": ["redis_tools", "slack_tools"],  # Include specific shared tools
    "shared": [],  # No shared tools
}
```

## 📊 **Role Examples**

### **Router Role (JSON Response)**

- **Tools**: None
- **Strategy**: JSON response with Pydantic parsing
- **Built-in tools**: Excluded (`include_builtin: false`)
- **Purpose**: Route requests to appropriate roles

### **Weather Role (Pre-Processing)**

- **Tools**: None
- **Strategy**: Pre-fetch weather data, inject into prompt
- **Built-in tools**: Excluded (`include_builtin: false`)
- **Purpose**: Provide weather information with external data

### **Timer Role (Tool-Based)**

- **Tools**: Custom timer tools (`set_timer`, `cancel_timer`, etc.)
- **Strategy**: LLM calls tools to perform timer operations
- **Built-in tools**: Excluded (`include_builtin: false`)
- **Purpose**: Manage timers and alarms

## 🎯 **Best Practices**

### **1. Single Responsibility**

- Each role should have one clear purpose
- All related functionality in one file
- Clear separation from other roles

### **2. LLM-Friendly Design**

- Consistent structure across all roles
- Clear documentation and comments
- Simple, understandable patterns

### **3. Tool Configuration**

- Explicitly configure tool requirements
- Use `include_builtin: false` for specialized roles
- Only include tools that the role actually needs

### **4. Error Handling**

- Comprehensive error handling in all functions
- Return intents for errors, don't raise exceptions
- Graceful degradation and fallback behavior

### **5. Testing**

- Write comprehensive tests for each role
- Test happy path, error cases, and edge cases
- Validate intent generation and tool execution

## 🔄 **Migration from Multi-File Roles**

To migrate a legacy multi-file role to single-file:

1. **Extract metadata** from `definition.yaml` → `ROLE_CONFIG` dict
2. **Extract handlers** from `lifecycle.py` → Pure functions returning intents
3. **Extract tools** from `tools.py` → `@tool` decorated functions
4. **Add intents** inline → `@dataclass` definitions
5. **Configure tools** → Add `tools` configuration to `ROLE_CONFIG`
6. **Test thoroughly** → Ensure all functionality works

## 📖 **Documentation**

For detailed implementation guidance, see:

- **[Document 25](../docs/25_THREADING_ARCHITECTURE_IMPROVEMENTS.md)** - Low-level implementation details
- **[Document 26](../docs/26_HIGH_LEVEL_ARCHITECTURE_PATTERNS.md)** - High-level architecture patterns
- **[Document 27](../docs/27_THREADING_FIX_IMPLEMENTATION_PLAN.md)** - Implementation plan

## 🎯 **Role Development Checklist**

When creating a new role:

- [ ] Follow single-file pattern with 10 sections
- [ ] Add proper `ROLE_CONFIG` with tool configuration
- [ ] Implement pure function event handlers
- [ ] Add role-specific intents if needed
- [ ] Configure tools appropriately (`include_builtin`, `automatic`, `shared`)
- [ ] Add comprehensive error handling
- [ ] Write tests for all functionality
- [ ] Document the role's purpose and usage
- [ ] Validate with role registry loading
- [ ] Test integration with UniversalAgent

This architecture ensures that roles are **LLM-safe**, **maintainable**, and **perfectly aligned** with the system's design principles.
