# Pure Intent Architecture Refactor Design

**Document ID:** 29
**Created:** 2025-10-14
**Status:** Architectural Purity Analysis and Refactor Design
**Priority:** High
**Context:** Compliance with Documents 25, 26, 27 Architectural Principles
**Companion:** Document 28 - Current Implementation Analysis

## Executive Summary

After implementing the intent-based timer architecture, critical reflection reveals **architectural violations** against the pure patterns defined in Documents 25, 26, 27. While the current implementation **works functionally**, it violates core separation of concerns principles.

## Current Implementation Analysis

### ✅ **What We Got Right:**

1. **Declarative Tools**: Tools return intents instead of doing I/O
2. **Intent-Based Processing**: Separated "what" from "how"
3. **Single Event Loop**: No threading.Timer usage
4. **Context Flow**: RequestMetadata → LLMSafeEventContext → Intent context
5. **Functional Success**: Timer notifications route correctly

### ❌ **Architectural Violations Identified:**

#### **Violation 1: Intent Handlers in Roles (Separation of Concerns)**

```python
# CURRENT (WRONG) - In roles/timer_single_file.py
async def process_timer_creation_intent(intent: TimerCreationIntent):
    # ❌ Role doing infrastructure I/O operations
    redis_write(f"timer:{intent.timer_id}", timer_data)
    asyncio.create_task(_schedule_timer_expiry_async(...))
```

**Document 26 Principle**: Roles should focus on business logic, infrastructure handles I/O.

#### **Violation 2: Complex Context Injection (LLM-Friendly Simplicity)**

```python
# CURRENT (COMPLEX) - In llm_provider/universal_agent.py
class IntentProcessingHook(HookProvider):
    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(AfterToolCallEvent, self._process_tool_result_intents)
```

**Document 26 Principle**: Create the simplest possible patterns that AI agents can follow.

#### **Violation 3: Tools Return Complex Dictionaries**

```python
# CURRENT (COMPLEX) - Tools return dict with intent data
return {
    "success": True,
    "intent": {
        "type": "TimerCreationIntent",  # String type, not object
        "timer_id": timer_id,
        # ... complex dictionary structure
    }
}
```

**Document 26 Template**: Tools should return intent objects directly.

## Pure Architecture Refactor Design

### **1. Simplified Timer Role (Business Logic Only)**

```python
# roles/timer_single_file.py - PURE VERSION
"""Timer role - Pure business logic, no I/O operations."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from common.intents import Intent, NotificationIntent, AuditIntent
from strands import tool

# 1. ROLE METADATA
ROLE_CONFIG = {
    "name": "timer",
    "version": "6.0.0",  # Pure architecture version
    "description": "Timer management with pure intent-based architecture",
    "llm_type": "WEAK",
    "fast_reply": True,
}

# 2. ROLE-SPECIFIC INTENTS (owned by timer role)
@dataclass
class TimerCreationIntent(Intent):
    """Timer creation intent - pure data structure."""
    timer_id: str
    duration: str
    duration_seconds: int
    label: str = ""
    user_id: Optional[str] = None
    channel_id: Optional[str] = None

    def validate(self) -> bool:
        return bool(self.timer_id and self.duration and self.duration_seconds > 0)

@dataclass
class TimerCancellationIntent(Intent):
    """Timer cancellation intent - pure data structure."""
    timer_id: str
    user_id: Optional[str] = None

    def validate(self) -> bool:
        return bool(self.timer_id)

# 3. EVENT HANDLERS (pure functions returning intents)
def handle_timer_expiry(event_data: Any, context: LLMSafeEventContext) -> List[Intent]:
    """LLM-SAFE: Pure function - no I/O operations."""
    timer_id, request = _parse_timer_event_data(event_data)

    return [
        NotificationIntent(
            message=f"⏰ Timer expired: {request}",
            channel=context.get_safe_channel(),
            user_id=context.user_id,
            priority="medium",
        ),
        AuditIntent(
            action="timer_expired",
            details={"timer_id": timer_id, "original_request": request},
            user_id=context.user_id,
        ),
    ]

# 4. TOOLS (pure business logic, return intent objects)
@tool
def set_timer(duration: str, label: str = "") -> Dict[str, Any]:
    """LLM-SAFE: Pure business logic - returns intent object."""
    try:
        duration_seconds = _parse_duration(duration)
        if duration_seconds <= 0:
            return {"success": False, "error": f"Invalid duration: {duration}"}

        timer_id = f"timer_{uuid.uuid4().hex[:8]}"

        # PURE: Return intent object directly (not dictionary)
        return {
            "success": True,
            "message": f"Timer set for {duration}" + (f" ({label})" if label else ""),
            "intent": TimerCreationIntent(
                timer_id=timer_id,
                duration=duration,
                duration_seconds=duration_seconds,
                label=label,
                # Context will be injected by infrastructure
            )
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# 5. HELPER FUNCTIONS (pure business logic only)
def _parse_duration(duration_str: str) -> int:
    """Pure function - no side effects."""
    # ... parsing logic (no I/O)

def _parse_timer_event_data(event_data: Any) -> tuple[str, str]:
    """Pure function - no side effects."""
    # ... parsing logic (no I/O)

# 6. ROLE REGISTRATION (no intent handlers - handled by infrastructure)
def register_role():
    """Auto-discovered by RoleRegistry."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "TIMER_EXPIRED": handle_timer_expiry,
        },
        "tools": [set_timer, cancel_timer, list_timers],
        "intents": [TimerCreationIntent, TimerCancellationIntent, TimerListingIntent],
        # ❌ NO intent handlers - these belong in infrastructure
    }
```

### **2. Infrastructure Handles All I/O (IntentProcessor)**

```python
# common/intent_processor.py - PURE VERSION
"""Intent processor handles ALL I/O operations."""

import asyncio
import time
from roles.timer_single_file import TimerCreationIntent, TimerCancellationIntent

class IntentProcessor:
    def __init__(self, communication_manager=None, workflow_engine=None, message_bus=None):
        self.communication_manager = communication_manager
        self.workflow_engine = workflow_engine
        self.message_bus = message_bus

        # Core intent handlers (infrastructure responsibility)
        self._core_handlers = {
            NotificationIntent: self._process_notification,
            AuditIntent: self._process_audit,
            WorkflowIntent: self._process_workflow,
            ErrorIntent: self._process_error,
            # Timer intents handled by infrastructure
            TimerCreationIntent: self._process_timer_creation,
            TimerCancellationIntent: self._process_timer_cancellation,
            TimerListingIntent: self._process_timer_listing,
        }

    async def _process_timer_creation(self, intent: TimerCreationIntent):
        """Infrastructure handles timer I/O operations."""
        from roles.shared_tools.redis_tools import redis_write

        # Create timer data with context
        timer_data = {
            "id": intent.timer_id,
            "duration": intent.duration,
            "duration_seconds": intent.duration_seconds,
            "label": intent.label,
            "created_at": time.time(),
            "expires_at": time.time() + intent.duration_seconds,
            "status": "active",
            "user_id": intent.user_id,      # From context injection
            "channel_id": intent.channel_id, # From context injection
        }

        # Store in Redis (infrastructure responsibility)
        redis_result = redis_write(
            f"timer:{intent.timer_id}",
            timer_data,
            ttl=intent.duration_seconds + 60
        )

        if redis_result.get("success"):
            # Schedule expiry (infrastructure responsibility)
            asyncio.create_task(
                self._schedule_timer_expiry(intent.timer_id, intent.duration_seconds, timer_data)
            )

    async def _schedule_timer_expiry(self, timer_id: str, duration_seconds: int, timer_data: dict):
        """Infrastructure schedules timer expiry."""
        await asyncio.sleep(duration_seconds)

        # Emit timer expiry event
        self.message_bus.publish(
            self,
            "TIMER_EXPIRED",
            {
                "timer_id": timer_id,
                "original_request": f"Timer {timer_data.get('duration')} expired",
                "user_id": timer_data.get("user_id"),
                "channel_id": timer_data.get("channel_id"),
            }
        )

        # Clean up
        from roles.shared_tools.redis_tools import redis_delete
        redis_delete(f"timer:{timer_id}")
```

### **3. Simplified Context Injection (Infrastructure)**

```python
# In common/intent_processor.py or WorkflowEngine
def inject_context_into_intent(intent: Intent, context: LLMSafeEventContext) -> Intent:
    """Simple context injection - no complex hooks."""
    if hasattr(intent, 'user_id'):
        intent.user_id = context.user_id
    if hasattr(intent, 'channel_id'):
        intent.channel_id = context.channel_id
    return intent

# In UniversalAgent.execute_task()
def process_tool_results_with_context(tool_results, context):
    """Simple tool result processing."""
    for result in tool_results:
        if 'intent' in result and isinstance(result['intent'], Intent):
            # Inject context into intent object
            intent_with_context = inject_context_into_intent(result['intent'], context)
            # Process through IntentProcessor
            await self.intent_processor.process_intents([intent_with_context])
```

### **4. Clean Tool Result Processing (No Hooks)**

```python
# In llm_provider/universal_agent.py - SIMPLIFIED
def execute_task(self, instruction: str, role: str, event_context: LLMSafeEventContext = None):
    # Execute with Strands agent
    response = agent(instruction)

    # Simple tool result processing (no hooks)
    if hasattr(response, 'tool_results') and event_context:
        for tool_result in response.tool_results:
            if isinstance(tool_result, dict) and 'intent' in tool_result:
                intent = tool_result['intent']
                if isinstance(intent, Intent):
                    # Simple context injection
                    intent = self._inject_context(intent, event_context)
                    # Process through infrastructure
                    await self.intent_processor.process_intents([intent])

    return self._extract_response_text(response)
```

## Architecture Comparison

### **Current Implementation (Functional but Impure)**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Timer Role      │    │ UniversalAgent   │    │ IntentProcessor │
│ - Tools         │    │ - Strands Hooks  │    │ - Registration  │
│ - Intents       │ ←→ │ - Context Inject │ ←→ │ - No I/O        │
│ - Intent I/O ❌ │    │ - Complex ❌     │    │ - Simple ✅     │
│ - Redis Ops ❌  │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Pure Architecture (Document 26 Compliant)**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Timer Role      │    │ UniversalAgent   │    │ IntentProcessor │
│ - Tools ✅      │    │ - Simple Inject  │    │ - All I/O Ops   │
│ - Intents ✅    │ →  │ - No Hooks ✅    │ →  │ - Redis ✅      │
│ - Pure Logic ✅ │    │ - Clean ✅       │    │ - Asyncio ✅    │
│ - No I/O ✅     │    │                  │    │ - Scheduling ✅ │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Refactor Benefits

### **1. True Separation of Concerns**

- **Roles**: Pure business logic, no I/O operations
- **Infrastructure**: All I/O, scheduling, and system operations
- **Clear Boundaries**: No mixing of concerns

### **2. LLM-Friendly Simplicity**

- **No Complex Hooks**: Simple function calls for context injection
- **Clear Patterns**: Easy for AI agents to understand and modify
- **Predictable Flow**: Linear processing, no callback complexity

### **3. Architectural Purity**

- **Document 26 Compliance**: Follows template exactly
- **Intent Ownership**: Roles own intents, infrastructure processes them
- **Pure Functions**: All role functions have no side effects

### **4. Maintainability**

- **Single Responsibility**: Each component has one clear purpose
- **Testability**: Easy to test business logic separately from I/O
- **Extensibility**: Easy to add new intent types and handlers

## Refactor Implementation Plan

### **Phase 1: Move Intent Handlers to Infrastructure** (45 minutes)

1. **Move** `process_timer_*_intent` functions from timer role to `IntentProcessor`
2. **Update** `IntentProcessor` to handle timer intents directly
3. **Remove** intent handler registration from timer role

### **Phase 2: Simplify Context Injection** (30 minutes)

1. **Remove** Strands hook system from `UniversalAgent`
2. **Implement** simple context injection in `execute_task`
3. **Update** tools to return intent objects (not dictionaries)

### **Phase 3: Clean Tool Result Processing** (30 minutes)

1. **Replace** hook-based processing with direct tool result inspection
2. **Simplify** intent extraction and processing
3. **Remove** complex callback mechanisms

### **Phase 4: Validation and Testing** (30 minutes)

1. **Test** that context injection still works
2. **Validate** architectural purity compliance
3. **Ensure** functional behavior is preserved

## Architecture Principles Compliance

### **Document 25 Principles:**

- ✅ **Single Event Loop**: Maintained
- ✅ **No Threading**: Maintained
- ❌ **Simplicity**: Current implementation too complex

### **Document 26 Principles:**

- ✅ **Intent-Based**: Maintained
- ❌ **Separation of Concerns**: Violated (roles do I/O)
- ❌ **LLM-Friendly**: Complex hooks violate simplicity

### **Document 27 Principles:**

- ✅ **Pure Functions**: Tools are pure
- ❌ **Infrastructure Responsibility**: Roles handle infrastructure concerns

## Recommended Action

**Refactor to Pure Architecture** for the following reasons:

1. **Architectural Integrity**: Maintain consistency with system design principles
2. **Future Maintainability**: Other roles should follow the same pure pattern
3. **LLM Development**: Simpler patterns are easier for AI agents to work with
4. **Technical Debt**: Current violations will compound over time

## Implementation Effort

- **Time**: ~2.5 hours total
- **Risk**: Low (functional behavior preserved)
- **Benefit**: High (architectural purity, maintainability)
- **Complexity**: Medium (requires careful refactoring)

## Success Criteria for Pure Architecture

1. **✅ Roles contain zero I/O operations**
2. **✅ IntentProcessor handles all timer I/O**
3. **✅ Context injection is simple and direct**
4. **✅ No Strands hooks or complex callbacks**
5. **✅ Tools return intent objects, not dictionaries**
6. **✅ Functional behavior preserved (context routing works)**

The pure architecture would maintain all functional benefits while achieving true architectural compliance with the LLM-safe design principles.
