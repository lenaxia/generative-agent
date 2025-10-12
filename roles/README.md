# LLM-Friendly Role Development Guide

**Context:** 100% LLM Development Environment
**Architecture:** Single-File Role Pattern
**Updated:** 2025-10-12

## Overview

This directory contains roles for the Universal Agent System, designed specifically for LLM development. Each role is contained in a single Python file following consistent patterns that AI agents can reliably understand and extend.

## Single-File Role Architecture

### **Why Single Files?**

**For LLM Development:**

- **Single Context**: AI agents see entire role in one file
- **Clear Dependencies**: All imports visible at the top
- **Atomic Modifications**: Change entire role without affecting others
- **Easier Understanding**: Complete role logic in single context window

**For Maintenance:**

- **Reduced Complexity**: 1 file instead of 4+ files per role
- **Simple Navigation**: Everything in one place
- **Clear Ownership**: One file owns all role concerns
- **Easy Testing**: Single file to test

## Role File Template

Every role should follow this exact structure:

```python
"""Role description - LLM-friendly single file implementation."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time
import logging
from common.intents import Intent, NotificationIntent, AuditIntent
from strands import tool

logger = logging.getLogger(__name__)

# 1. ROLE METADATA (replaces definition.yaml)
ROLE_CONFIG = {
    "name": "role_name",
    "version": "1.0.0",
    "description": "What this role does",
    "llm_type": "WEAK|DEFAULT|STRONG",
    "fast_reply": True,
    "when_to_use": "When to use this role"
}

# 2. ROLE-SPECIFIC INTENTS (owned by this role)
@dataclass
class RoleSpecificIntent(Intent):
    """Role-specific intent - owned by this role only."""
    action: str
    parameters: Dict[str, Any]

    def validate(self) -> bool:
        return bool(self.action and isinstance(self.parameters, dict))

# 3. EVENT HANDLERS (pure functions returning intents)
def handle_role_event(event_data: Any, context) -> List[Intent]:
    """LLM-SAFE: Pure function event handler template."""
    try:
        # Step 1: Parse input data
        parsed_data = _parse_event_data(event_data)

        # Step 2: Create intents based on business logic
        intents = []

        # Always create notification
        intents.append(NotificationIntent(
            message=f"Event processed: {parsed_data}",
            channel=context.channel_id or "general",
            user_id=context.user_id
        ))

        # Always create audit
        intents.append(AuditIntent(
            action="event_handled",
            details={"data": parsed_data, "timestamp": time.time()},
            user_id=context.user_id
        ))

        # Step 3: Return intents (no side effects)
        return intents

    except Exception as e:
        logger.error(f"Event handler error: {e}")
        # Return error intent instead of raising
        return [
            NotificationIntent(
                message=f"Event processing error: {e}",
                channel=context.channel_id or "general",
                priority="high"
            )
        ]

# 4. TOOLS (simplified, returning intents or simple data)
@tool
def role_action(parameter: str) -> Dict[str, Any]:
    """LLM-SAFE: Simplified tool template."""
    try:
        return {
            "success": True,
            "message": f"Action completed: {parameter}",
            "data": {"parameter": parameter, "timestamp": time.time()}
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# 5. HELPER FUNCTIONS (minimal, focused)
def _parse_event_data(event_data: Any) -> Dict[str, Any]:
    """LLM-SAFE: Simple parsing that LLMs can understand."""
    try:
        if isinstance(event_data, dict):
            return event_data
        elif isinstance(event_data, list):
            return {"items": event_data}
        else:
            return {"raw": str(event_data)}
    except Exception as e:
        return {"error": str(e)}

# 6. INTENT HANDLER (processes role-specific intents)
async def process_role_intent(intent: RoleSpecificIntent):
    """Process role-specific intents - called by IntentProcessor."""
    logger.info(f"Processing {intent.action} with {intent.parameters}")
    # Implementation depends on role's specific needs

# 7. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "ROLE_EVENT": handle_role_event
        },
        "tools": [role_action],
        "intents": {
            RoleSpecificIntent: process_role_intent
        }
    }
```

## LLM Development Guidelines

### **✅ DO: LLM-Friendly Patterns**

1. **Explicit Dependencies**: All imports at the top, all parameters in function signatures
2. **Pure Functions**: Event handlers return intents, no side effects
3. **Clear Error Handling**: Try/except with explicit error intents
4. **Consistent Structure**: Follow the template exactly
5. **Simple Logic**: Keep functions focused and understandable

### **🚫 DON'T: LLM-Hostile Patterns**

1. **Hidden Dependencies**: No global variables or implicit context
2. **Complex Async**: No complex threading or event loop manipulation
3. **Side Effects**: No direct I/O operations in event handlers
4. **Magic Values**: No hardcoded constants, use configuration
5. **Complex Control Flow**: No nested callbacks or complex state machines

### **🔍 WATCH OUT FOR: Common LLM Pitfalls**

1. **Pattern Drift**: Gradually deviating from established structure
2. **Over-Abstraction**: Creating unnecessary complexity
3. **Missing Validation**: Forgetting to implement intent validation
4. **Silent Failures**: Not handling errors explicitly
5. **Implicit Context**: Assuming context that isn't visible

## Creating a New Role

### **Step 1: Create Role File**

```bash
# Create new role file
touch roles/my_new_role.py
```

### **Step 2: Copy Template**

Copy the role template above and modify:

- Change `role_name` to your role name
- Update `ROLE_CONFIG` with role details
- Define role-specific intents
- Implement event handlers
- Add tools as needed

### **Step 3: Test Role**

```python
# Test your role
from roles.my_new_role import register_role, ROLE_CONFIG

def test_new_role():
    role_info = register_role()
    assert "config" in role_info
    assert role_info["config"]["name"] == "my_new_role"
    print("✅ New role working")

test_new_role()
```

### **Step 4: Register with System**

The role will be automatically discovered by `RoleRegistry` when the system starts.

## Role Examples

### **Simple Role (LLM-Only)**

```python
# roles/simple_assistant.py
ROLE_CONFIG = {
    "name": "simple_assistant",
    "llm_type": "DEFAULT",
    "description": "General purpose assistant"
}

def register_role():
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {},  # No event handlers
        "tools": [],           # No custom tools
        "intents": {}          # No custom intents
    }
```

### **Complex Role (With Custom Logic)**

```python
# roles/weather.py
@dataclass
class WeatherIntent(Intent):
    location: str
    forecast_type: str = "current"

    def validate(self) -> bool:
        return bool(self.location)

def handle_weather_update(event_data, context) -> List[Intent]:
    return [
        WeatherIntent(location=event_data.get('location', 'unknown')),
        NotificationIntent(message="Weather updated")
    ]

@tool
def get_weather(location: str) -> Dict[str, Any]:
    return {
        "success": True,
        "intent": WeatherIntent(location=location)
    }

def register_role():
    return {
        "config": {"name": "weather", "llm_type": "WEAK"},
        "event_handlers": {"WEATHER_UPDATE": handle_weather_update},
        "tools": [get_weather],
        "intents": {WeatherIntent: process_weather_intent}
    }
```

## Testing Your Roles

### **Basic Role Test Template**

```python
# tests/test_my_role.py
def test_role_structure():
    from roles.my_role import register_role, ROLE_CONFIG

    # Test config
    assert "name" in ROLE_CONFIG
    assert "llm_type" in ROLE_CONFIG

    # Test registration
    role_info = register_role()
    assert "config" in role_info

def test_event_handlers():
    from roles.my_role import handle_my_event

    result = handle_my_event({"test": "data"}, mock_context)
    assert isinstance(result, list)
    assert all(hasattr(intent, 'validate') for intent in result)
```

## Migration from Multi-File Roles

### **Migration Steps**

1. **Extract Metadata**: Copy from `definition.yaml` to `ROLE_CONFIG`
2. **Extract Handlers**: Copy from `lifecycle.py`, make them pure functions
3. **Extract Tools**: Copy from `tools.py`, simplify to return intents
4. **Add Registration**: Create `register_role()` function
5. **Test**: Verify role works with new structure

### **Example Migration**

```python
# Before: roles/timer/definition.yaml + lifecycle.py + tools.py (1800+ lines)
# After: roles/timer.py (200 lines)

# Migration result:
ROLE_CONFIG = {  # From definition.yaml
    "name": "timer",
    "version": "3.0.0"
}

def handle_timer_expiry(event_data, context):  # From lifecycle.py
    return [NotificationIntent(...)]

@tool
def set_timer(duration: str):  # From tools.py
    return {"intent": TimerIntent(...)}
```

## Best Practices for LLM Development

### **Code Organization**

- Keep functions under 50 lines
- Use descriptive variable names
- Add type hints for all parameters
- Include docstrings for all functions

### **Error Handling**

- Always use try/except in event handlers
- Return error intents instead of raising exceptions
- Log errors with context information
- Provide fallback behavior

### **Testing**

- Test all event handlers with various input formats
- Validate all intents before processing
- Test error conditions and edge cases
- Use mock objects for dependencies

By following these guidelines, LLM agents can reliably create, modify, and extend roles while maintaining system stability and threading safety.
