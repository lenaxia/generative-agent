# High-Level Architecture Patterns for LLM-Driven Development

**Document ID:** 26
**Created:** 2025-10-12
**Status:** Strategic Architecture Design for AI Development
**Priority:** Strategic
**Context:** 100% LLM Development Environment
**Companion:** Document 25 - Low-Level Implementation Details

## Rules

- Regularly run `make lint` to validate that your code is healthy
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
- Do not use fallbacks. Fallbacks tend to be brittle and fragile. Do implement fallbacks of any kind.
- Whenever you complete a phase, make sure to update this checklist
- Don't just blindly implement changes. Reflect on them to make sure they make sense within the larger project. Pull in other files if additional context is needed
- When you complete the implementation of a project add new todo items addressing outstanding technical debt related to what you just implemented, such as removing old code, updating documentation, searching for additional references, etc. Fix these issues, do not accept technical debt for the project being implemented.

## Executive Summary

This document defines the high-level architectural strategy for eliminating threading issues through proper design patterns **specifically optimized for LLM-driven development**. The architecture emphasizes simplicity, predictability, and single-file role patterns that AI agents can reliably understand and extend.

**Core Principle for LLM Development:** Create the simplest possible architecture that eliminates entire classes of errors while providing clear, consistent patterns that AI agents can follow reliably.

## Simplified Architecture Strategy

### **Single Event Loop Foundation**

**Concept:** Eliminate all threading complexity by moving everything to the main event loop.

```
Current (Complex):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Main Thread │    │ Heartbeat   │    │ Timer       │
│ (Supervisor)│    │ Thread      │    │ Monitor     │
│             │    │             │    │ Thread      │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └─────── Cross-Thread Async Issues ────┘

Simplified (LLM-Safe):
┌─────────────────────────────────────────────────────┐
│              Single Event Loop                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │ Supervisor  │ │ Scheduled   │ │ Scheduled   │   │
│  │             │ │ Heartbeat   │ │ Timer Check │   │
│  │             │ │ Task        │ │ Task        │   │
│  └─────────────┘ └─────────────┘ └─────────────┘   │
└─────────────────────────────────────────────────────┘
```

### **Intent-Based Architecture**

**Concept:** Separate "what should happen" (intents) from "how to make it happen" (infrastructure).

```
Current (Imperative):
Event Handler → Direct I/O Operations → Side Effects

Simplified (Declarative):
Event Handler → Return Intents → Intent Processor → Side Effects
```

### **Single-File Role Architecture**

**Concept:** Each role is completely self-contained in a single Python file.

```
Current (Complex):
roles/timer/
├── definition.yaml (199 lines)
├── lifecycle.py (1201 lines)
├── tools.py (443 lines)
└── intents.py (proposed)

Simplified (LLM-Friendly):
roles/timer.py (200 lines total)
├── ROLE_CONFIG (metadata)
├── Role-specific intents
├── Event handlers (pure functions)
├── Tools (simplified)
└── Helper functions
```

## LLM-Optimized Design Patterns

### Pattern 1: Single-File Role Pattern

**Template Structure for All Roles:**

```python
# roles/{role_name}.py - Complete role in single file
"""Role description - LLM-friendly single file implementation."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from common.intents import Intent, NotificationIntent, AuditIntent
from strands import tool

# 1. ROLE METADATA (replaces definition.yaml)
ROLE_CONFIG = {
    "name": "role_name",
    "version": "1.0.0",
    "description": "Role description",
    "llm_type": "WEAK|DEFAULT|STRONG",
    "fast_reply": True,
    "when_to_use": "When to use this role"
}

# 2. ROLE-SPECIFIC INTENTS (owned by this role)
@dataclass
class RoleSpecificIntent(Intent):
    """Role-specific intent - owned by this role."""
    action: str
    parameters: Dict[str, Any]

    def validate(self) -> bool:
        return bool(self.action and isinstance(self.parameters, dict))

# 3. EVENT HANDLERS (pure functions returning intents)
def handle_role_event(event_data: Any, context) -> List[Intent]:
    """LLM-SAFE: Pure function event handler template."""
    # Parse input
    parsed_data = _parse_event_data(event_data)

    # Create intents
    return [
        NotificationIntent(message=f"Event processed: {parsed_data}"),
        AuditIntent(action="event_handled", details={"data": parsed_data})
    ]

# 4. TOOLS (simplified, returning intents or simple data)
@tool
def role_action(parameter: str) -> Dict[str, Any]:
    """LLM-SAFE: Simplified tool."""
    return {
        "success": True,
        "message": f"Action completed: {parameter}",
        "intent": RoleSpecificIntent(action="perform", parameters={"param": parameter})
    }

# 5. HELPER FUNCTIONS (minimal, focused)
def _parse_event_data(event_data: Any) -> Dict[str, Any]:
    """Simple parsing logic that LLMs can understand."""
    if isinstance(event_data, dict):
        return event_data
    elif isinstance(event_data, list):
        return {"items": event_data}
    else:
        return {"raw": str(event_data)}

# 6. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "ROLE_EVENT": handle_role_event
        },
        "tools": [role_action],
        "intents": [RoleSpecificIntent]
    }
```

### Pattern 2: Layered Intent Architecture

**Core System Intents (Universal):**

```python
# common/intents.py - ONLY universal intents
@dataclass
class NotificationIntent(Intent):
    """Universal: Any role can send notifications."""

@dataclass
class AuditIntent(Intent):
    """Universal: Any role can audit actions."""

@dataclass
class WorkflowIntent(Intent):
    """Universal: Any role can start workflows."""
```

**Role-Specific Intents (Owned by Roles):**

```python
# Inside roles/timer.py
@dataclass
class TimerIntent(Intent):
    """Timer-specific: Only timer role creates these."""

# Inside roles/weather.py
@dataclass
class WeatherIntent(Intent):
    """Weather-specific: Only weather role creates these."""
```

### Pattern 3: Pure Function Event Handlers

**LLM-Safe Event Handler Template:**

```python
def handle_{event_type}(event_data: Any, context) -> List[Intent]:
    """
    LLM-SAFE TEMPLATE:
    1. Parse input data explicitly
    2. Create intents declaratively
    3. Return results with no side effects
    """
    try:
        # Step 1: Parse and validate
        parsed_data = _parse_event_data(event_data)

        # Step 2: Business logic (pure functions only)
        intents = []
        if _should_notify(parsed_data):
            intents.append(NotificationIntent(...))
        if _should_audit(parsed_data):
            intents.append(AuditIntent(...))

        # Step 3: Return intents
        return intents

    except Exception as e:
        # LLM-SAFE: Return error intent instead of raising
        return [
            NotificationIntent(
                message=f"Event processing error: {e}",
                channel=context.channel_id or "general",
                priority="high"
            )
        ]
```

## LLM Development Benefits

### **Simplified Role Development**

**For AI Agents:**

1. **Single Context**: Entire role visible in one file
2. **Clear Dependencies**: All imports at the top
3. **Predictable Structure**: Same pattern for every role
4. **Easy Modification**: Change entire role atomically

**For System:**

1. **Reduced Complexity**: 1 file instead of 4+ files per role
2. **Clear Ownership**: Each role owns all its concerns
3. **Simple Discovery**: RoleRegistry finds single file per role
4. **Atomic Changes**: Modify role without affecting others

### **Intent Architecture Benefits**

**Separation of Concerns:**

- **Core System**: Universal intents only (Notification, Audit, Workflow)
- **Roles**: Own their specific intents and processing logic
- **Infrastructure**: Handles complex I/O operations

**Extensibility:**

- New roles add intents without modifying core system
- Dynamic registration supports any intent type
- LLMs can create new roles following established patterns

## Migration Strategy

### **Phase 1: Role Consolidation (Week 1)**

**Goal:** Consolidate existing multi-file roles into single files.

```
Migration Pattern:
roles/timer/ (4 files) → roles/timer.py (1 file)
├── Extract metadata from definition.yaml → ROLE_CONFIG dict
├── Extract handlers from lifecycle.py → Pure functions
├── Extract tools from tools.py → Simplified @tool functions
└── Add intents inline → @dataclass definitions
```

### **Phase 2: Pattern Standardization (Week 2)**

**Goal:** Ensure all roles follow the same LLM-friendly pattern.

```
Standardization:
├── Same file structure for all roles
├── Same function naming conventions
├── Same intent patterns
└── Same error handling approach
```

### **Phase 3: LLM Optimization (Week 3-4)**

**Goal:** Optimize patterns for AI development.

```
Optimization:
├── Add comprehensive documentation
├── Create role templates
├── Add validation frameworks
└── Implement testing patterns
```

## Configuration for Simplified Architecture

```yaml
# config.yaml - Simplified configuration
architecture:
  threading_model: "single_event_loop"
  role_system: "single_file"
  llm_development: true

role_system:
  # Single file role discovery
  roles_directory: "roles"
  role_pattern: "*.py" # Look for Python files, not directories

  # LLM development features
  auto_discovery: true
  validate_role_structure: true
  enforce_patterns: true

# Intent processing
intent_processing:
  enabled: true
  validate_intents: true
  timeout_seconds: 30
```

## Success Metrics

### **Simplification Metrics**

- **Files per Role**: Target 1 file (currently 3-4 files)
- **Lines per Role**: Target <300 lines (currently >1800 lines)
- **Dependencies**: Minimal imports, self-contained

### **LLM Development Metrics**

- **Pattern Consistency**: All roles follow same structure
- **Modification Success**: LLMs can successfully modify roles
- **Extension Success**: LLMs can create new roles following patterns

By implementing this simplified architecture, we create a system that is both thread-safe and LLM-safe, enabling effective AI-driven development while eliminating the complexity that makes the current system difficult to understand and maintain.
