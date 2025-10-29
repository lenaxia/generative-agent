# Timer Context Routing Architecture Design

**Document ID:** 28
**Created:** 2025-10-13
**Status:** LLM Implementation Guide
**Priority:** High
**Context:** Intent-Based Architecture for Timer Context Routing
**Companion:** Documents 25, 26, 27 - Threading Architecture Improvements

## Executive Summary

This document provides the **architecturally correct** solution for timer context routing that follows the LLM-safe intent-based architecture. The previous approach of using Strands ToolContext injection was **architecturally incorrect** and violated core design principles.

## Architectural Problem Analysis

### ❌ **Current Incorrect Implementation**

The timer role currently uses **imperative tools** that perform direct I/O operations:

```python
@tool
def set_timer(duration: str, label: str = "") -> dict[str, Any]:
    # WRONG: Direct I/O operations (imperative pattern)
    timer_data = {
        "user_id": "system",    # ❌ HARDCODED - violates context flow
        "channel": "console",   # ❌ HARDCODED - violates context flow
    }
    redis_write(f"timer:{timer_id}", timer_data)  # ❌ Direct side effect
    threading.Timer(...).start()                 # ❌ Direct side effect
    return {"success": True}
```

**Architectural Violations:**

1. **Imperative Pattern**: Tools perform side effects directly
2. **Hardcoded Context**: No access to request metadata
3. **Threading Complexity**: Uses `threading.Timer` (violates single event loop)
4. **Tight Coupling**: Tools directly access Redis and system resources

### ❌ **Previous Incorrect Solution Approach**

The initial solution of using Strands ToolContext injection was also architecturally wrong:

```python
# WRONG APPROACH - Violates LLM-safe architecture
@tool(context=True)
def set_timer(duration: str, tool_context: ToolContext) -> dict[str, Any]:
    # Still imperative, still direct I/O, wrong context type
```

**Why This Was Wrong:**

1. **Mixed Context Types**: Strands ToolContext ≠ LLMSafeEventContext
2. **Still Imperative**: Tools still perform direct I/O operations
3. **Framework Coupling**: Tight coupling to Strands framework internals
4. **Violates Intent Pattern**: Doesn't follow "Return Intents → Intent Processor" pattern

## ✅ **Correct LLM-Safe Intent-Based Architecture**

### **Core Principle from Document 26:**

```
Event Handler → Return Intents → Intent Processor → Side Effects
```

### **Correct Implementation Pattern**

#### **1. Declarative Timer Tools**

```python
@tool
def set_timer(duration: str, label: str = "") -> dict[str, Any]:
    """LLM-SAFE: Declarative timer creation - returns intent, no side effects."""
    try:
        # Validate input
        duration_seconds = _parse_duration(duration)
        if duration_seconds <= 0:
            return {"success": False, "error": f"Invalid duration: {duration}"}

        # Generate timer ID
        timer_id = f"timer_{uuid.uuid4().hex[:8]}"

        # CORRECT: Return intent instead of doing I/O
        return {
            "success": True,
            "timer_id": timer_id,
            "message": f"Timer set for {duration}" + (f" ({label})" if label else ""),
            "intent": {
                "type": "TimerCreationIntent",
                "duration": duration,
                "duration_seconds": duration_seconds,
                "label": label,
                "timer_id": timer_id,
                # Context will be injected by UniversalAgent
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
```

#### **2. Timer-Specific Intents**

```python
@dataclass
class TimerCreationIntent(Intent):
    """Intent to create a timer with proper context routing."""

    timer_id: str
    duration: str
    duration_seconds: int
    label: str = ""
    user_id: Optional[str] = None
    channel_id: Optional[str] = None

    def validate(self) -> bool:
        return (
            bool(self.timer_id and self.duration)
            and self.duration_seconds > 0
        )

@dataclass
class TimerExpiryIntent(Intent):
    """Intent for timer expiry notifications."""

    timer_id: str
    original_duration: str
    label: str = ""
    user_id: Optional[str] = None
    channel_id: Optional[str] = None

    def validate(self) -> bool:
        return bool(self.timer_id)
```

#### **3. Intent Processor Handles I/O Operations**

```python
# In common/intent_processor.py
async def _process_timer_creation(self, intent: TimerCreationIntent):
    """Process timer creation intent - handles actual I/O operations."""
    try:
        # Create timer data with proper context
        timer_data = {
            "id": intent.timer_id,
            "duration": intent.duration,
            "duration_seconds": intent.duration_seconds,
            "label": intent.label,
            "created_at": time.time(),
            "expires_at": time.time() + intent.duration_seconds,
            "status": "active",
            "user_id": intent.user_id,      # ✅ From intent context
            "channel_id": intent.channel_id, # ✅ From intent context
        }

        # Store in Redis
        redis_result = redis_write(
            f"timer:{intent.timer_id}",
            timer_data,
            ttl=intent.duration_seconds + 60
        )

        if redis_result.get("success"):
            # Schedule expiry using single event loop (not threading.Timer)
            asyncio.create_task(
                self._schedule_timer_expiry(intent.timer_id, intent.duration_seconds)
            )
            logger.info(f"Timer {intent.timer_id} created and scheduled")
        else:
            logger.error(f"Failed to store timer: {redis_result.get('error')}")

    except Exception as e:
        logger.error(f"Timer creation failed: {e}")

async def _schedule_timer_expiry(self, timer_id: str, duration_seconds: int):
    """Schedule timer expiry using asyncio (single event loop)."""
    await asyncio.sleep(duration_seconds)

    # Emit timer expiry event
    self.message_bus.publish(
        self,
        MessageType.TIMER_EXPIRED,
        {
            "timer_id": timer_id,
            "expired_at": time.time(),
        }
    )
```

## Context Flow Architecture

### **1. Request Ingestion**

```
Slack Message → CommunicationManager._handle_channel_message()
→ Creates RequestMetadata with user_id, channel_id
→ MessageBus.publish(INCOMING_REQUEST, request_metadata)
```

### **2. Workflow Processing**

```
WorkflowEngine.handle_request(request_metadata)
→ Creates LLMSafeEventContext from RequestMetadata
→ UniversalAgent.execute_task(instruction, role, context=event_context)
```

### **3. Tool Execution with Context Injection**

```python
# In UniversalAgent.execute_task()
def execute_task(self, instruction: str, role: str, context: LLMSafeEventContext):
    # Store context for tool access
    self._current_context = context

    # Execute with Strands agent
    response = agent(instruction)

    # Process any intents returned by tools
    if hasattr(response, 'tool_results'):
        for tool_result in response.tool_results:
            if 'intent' in tool_result:
                # Inject context into intent
                intent_data = tool_result['intent']
                intent_data['user_id'] = context.user_id
                intent_data['channel_id'] = context.channel_id

                # Create intent object and process
                intent = self._create_intent_from_data(intent_data)
                await self.intent_processor.process_intents([intent])
```

### **4. Intent Processing**

```
Tool returns TimerCreationIntent → IntentProcessor._process_timer_creation()
→ Redis storage with proper context → asyncio.create_task(timer_expiry)
```

### **5. Timer Expiry**

```
asyncio timer expires → MessageBus.emit(TIMER_EXPIRED)
→ handle_timer_expiry() → Returns NotificationIntent with stored context
→ IntentProcessor → CommunicationManager → Correct channel delivery
```

## Implementation Steps

### **Phase 1: Create Timer Intents**

**File**: `common/intents.py`
**Action**: Add timer-specific intents

```python
@dataclass
class TimerCreationIntent(Intent):
    """Intent to create a timer with context routing."""
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
    """Intent to cancel a timer."""
    timer_id: str
    user_id: Optional[str] = None

    def validate(self) -> bool:
        return bool(self.timer_id)

@dataclass
class TimerExpiryIntent(Intent):
    """Intent for timer expiry notifications."""
    timer_id: str
    original_duration: str
    label: str = ""
    user_id: Optional[str] = None
    channel_id: Optional[str] = None

    def validate(self) -> bool:
        return bool(self.timer_id)
```

### **Phase 2: Update Timer Tools to Be Declarative**

**File**: `roles/timer_single_file.py`
**Action**: Convert tools from imperative to declarative

```python
@tool
def set_timer(duration: str, label: str = "") -> dict[str, Any]:
    """LLM-SAFE: Declarative timer creation - returns intent, no side effects."""
    try:
        # Validate duration
        duration_seconds = _parse_duration(duration)
        if duration_seconds <= 0:
            return {"success": False, "error": f"Invalid duration: {duration}"}

        # Generate timer ID
        timer_id = f"timer_{uuid.uuid4().hex[:8]}"

        # CORRECT: Return intent data, no I/O operations
        return {
            "success": True,
            "timer_id": timer_id,
            "message": f"Timer set for {duration}" + (f" ({label})" if label else ""),
            "intent": {
                "type": "TimerCreationIntent",
                "timer_id": timer_id,
                "duration": duration,
                "duration_seconds": duration_seconds,
                "label": label,
                # user_id and channel_id will be injected by UniversalAgent
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool
def cancel_timer(timer_id: str) -> dict[str, Any]:
    """LLM-SAFE: Declarative timer cancellation - returns intent."""
    return {
        "success": True,
        "message": f"Timer {timer_id} cancelled",
        "intent": {
            "type": "TimerCancellationIntent",
            "timer_id": timer_id,
        }
    }

@tool
def list_timers() -> dict[str, Any]:
    """LLM-SAFE: Declarative timer listing - returns intent."""
    return {
        "success": True,
        "message": "Listing active timers",
        "intent": {
            "type": "TimerListingIntent",
        }
    }
```

### **Phase 3: Enhance UniversalAgent for Context Injection**

**File**: `llm_provider/universal_agent.py`
**Action**: Inject LLMSafeEventContext into tool results

```python
def execute_task(
    self,
    instruction: str,
    role: str = "default",
    llm_type: LLMType = LLMType.DEFAULT,
    context: Optional[LLMSafeEventContext] = None,  # Accept LLMSafeEventContext
) -> str:
    """Execute task with context injection for intent processing."""

    # Store context for tool result processing
    self._current_context = context

    # Execute with Strands agent (unchanged)
    agent = self.assume_role(role, llm_type, context)
    response = agent(instruction)

    # NEW: Process tool results for intents
    await self._process_tool_intents(response, context)

    return self._extract_response_text(response)

async def _process_tool_intents(self, response, context: Optional[LLMSafeEventContext]):
    """Process intents returned by tools."""
    if not hasattr(response, 'tool_results') or not context:
        return

    for tool_result in response.tool_results:
        if isinstance(tool_result, dict) and 'intent' in tool_result:
            intent_data = tool_result['intent']

            # Inject context into intent
            intent_data['user_id'] = context.user_id
            intent_data['channel_id'] = context.channel_id

            # Create and process intent
            intent = self._create_intent_from_data(intent_data)
            if intent and hasattr(self, 'intent_processor'):
                await self.intent_processor.process_intents([intent])

def _create_intent_from_data(self, intent_data: dict) -> Optional[Intent]:
    """Create intent object from tool result data."""
    intent_type = intent_data.get('type')

    if intent_type == 'TimerCreationIntent':
        return TimerCreationIntent(**{k: v for k, v in intent_data.items() if k != 'type'})
    elif intent_type == 'TimerCancellationIntent':
        return TimerCancellationIntent(**{k: v for k, v in intent_data.items() if k != 'type'})
    # ... other intent types

    return None
```

### **Phase 4: Enhance IntentProcessor for Timer Operations**

**File**: `common/intent_processor.py`
**Action**: Add timer intent handlers

```python
def __init__(self, communication_manager=None, workflow_engine=None):
    # ... existing code ...

    # Add timer intent handlers
    self._core_handlers.update({
        TimerCreationIntent: self._process_timer_creation,
        TimerCancellationIntent: self._process_timer_cancellation,
        TimerExpiryIntent: self._process_timer_expiry,
    })

async def _process_timer_creation(self, intent: TimerCreationIntent):
    """Process timer creation intent - handles actual I/O operations."""
    try:
        # Create timer data with proper context
        timer_data = {
            "id": intent.timer_id,
            "duration": intent.duration,
            "duration_seconds": intent.duration_seconds,
            "label": intent.label,
            "created_at": time.time(),
            "expires_at": time.time() + intent.duration_seconds,
            "status": "active",
            "user_id": intent.user_id,      # ✅ From intent context
            "channel_id": intent.channel_id, # ✅ From intent context
        }

        # Store in Redis
        from roles.shared_tools.redis_tools import redis_write
        redis_result = redis_write(
            f"timer:{intent.timer_id}",
            timer_data,
            ttl=intent.duration_seconds + 60
        )

        if redis_result.get("success"):
            # Schedule expiry using single event loop (not threading.Timer)
            asyncio.create_task(
                self._schedule_timer_expiry(intent.timer_id, intent.duration_seconds, timer_data)
            )
            logger.info(f"Timer {intent.timer_id} created and scheduled")
        else:
            logger.error(f"Failed to store timer: {redis_result.get('error')}")

    except Exception as e:
        logger.error(f"Timer creation failed: {e}")

async def _schedule_timer_expiry(self, timer_id: str, duration_seconds: int, timer_data: dict):
    """Schedule timer expiry using asyncio (single event loop)."""
    await asyncio.sleep(duration_seconds)

    # Create timer expiry event with stored context
    self.message_bus.publish(
        self,
        MessageType.TIMER_EXPIRED,
        {
            "timer_id": timer_id,
            "original_request": f"Timer {timer_data.get('duration', 'unknown')} expired",
            "user_id": timer_data.get("user_id"),
            "channel_id": timer_data.get("channel_id"),
            "label": timer_data.get("label", ""),
            "expired_at": time.time(),
        }
    )

    # Clean up expired timer
    from roles.shared_tools.redis_tools import redis_delete
    redis_delete(f"timer:{timer_id}")
```

### **Phase 5: Update WorkflowEngine for Context Creation**

**File**: `supervisor/workflow_engine.py`
**Action**: Create LLMSafeEventContext from RequestMetadata

```python
def _handle_fast_reply(self, request: RequestMetadata, routing_result: dict) -> str:
    """Execute fast-reply with proper context creation."""
    try:
        request_id = "fr_" + str(uuid.uuid4()).split("-")[-1]
        role = routing_result["route"]

        # Create LLMSafeEventContext from RequestMetadata
        from common.event_context import LLMSafeEventContext

        event_context = LLMSafeEventContext(
            user_id=request.metadata.get("user_id") if request.metadata else None,
            channel_id=request.metadata.get("channel_id") if request.metadata else None,
            source=request.source_id,
            metadata=request.metadata or {},
        )

        # Execute with proper context
        result = self.universal_agent.execute_task(
            instruction=request.prompt,
            role=role,
            context=event_context,  # Pass LLMSafeEventContext
        )

        # ... rest unchanged ...
```

## Context Flow Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Slack Message   │    │ RequestMetadata  │    │ LLMSafeEvent    │
│ user_id: U123   │ →  │ metadata: {      │ →  │ Context         │
│ channel: C456   │    │   user_id: U123  │    │ user_id: U123   │
│                 │    │   channel_id:... │    │ channel_id:...  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Timer Expiry    │    │ TimerCreation    │    │ Timer Tool      │
│ Notification    │ ←  │ Intent           │ ←  │ Returns Intent  │
│ → Correct       │    │ user_id: U123    │    │ (Declarative)   │
│   Channel       │    │ channel_id:...   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Benefits of Intent-Based Approach

### **1. LLM-Safe Architecture**

- **Pure Functions**: Tools have no side effects
- **Declarative**: Tools describe "what" not "how"
- **Testable**: Easy to test intent creation vs I/O operations

### **2. Single Event Loop Compliance**

- **No Threading**: Uses `asyncio.create_task()` instead of `threading.Timer`
- **Predictable**: All operations in main event loop
- **No Race Conditions**: No cross-thread async issues

### **3. Proper Context Flow**

- **Consistent**: Uses LLMSafeEventContext throughout
- **Traceable**: Clear context propagation path
- **Maintainable**: Single context type, single flow pattern

### **4. Separation of Concerns**

- **Tools**: Focus on business logic and validation
- **Intent Processor**: Handles all I/O and system operations
- **Event Handlers**: Pure functions returning intents

## Migration Strategy

### **Step 1: Add Timer Intents** (30 minutes)

- Add `TimerCreationIntent`, `TimerCancellationIntent`, `TimerExpiryIntent` to `common/intents.py`

### **Step 2: Update Timer Tools** (45 minutes)

- Convert `set_timer()`, `cancel_timer()`, `list_timers()` to return intents
- Remove all direct I/O operations from tools

### **Step 3: Enhance IntentProcessor** (60 minutes)

- Add timer intent handlers to `common/intent_processor.py`
- Implement `_process_timer_creation()`, `_process_timer_cancellation()`
- Replace `threading.Timer` with `asyncio.create_task()`

### **Step 4: Update UniversalAgent** (45 minutes)

- Add context injection for tool results
- Add intent processing pipeline
- Create `LLMSafeEventContext` from `RequestMetadata`

### **Step 5: Update WorkflowEngine** (30 minutes)

- Pass `LLMSafeEventContext` to `UniversalAgent.execute_task()`
- Remove individual field extraction

### **Step 6: Testing and Validation** (60 minutes)

- Test end-to-end timer creation with Slack context
- Verify timer expiry notifications route correctly
- Validate single event loop compliance

## Success Criteria

1. **✅ Timer notifications route to correct Slack channel 100% of the time**
2. **✅ No hardcoded "system" or "console" values in timer data**
3. **✅ No threading.Timer usage (single event loop compliance)**
4. **✅ Tools are pure functions (no direct I/O operations)**
5. **✅ Context flows through LLMSafeEventContext consistently**
6. **✅ Intent-based architecture pattern followed correctly**

## Architecture Compliance

This design follows all core principles from Documents 25, 26:

- ✅ **Single Event Loop**: No background threads, uses `asyncio.create_task()`
- ✅ **Intent-Based Processing**: Tools return intents, IntentProcessor handles I/O
- ✅ **LLM-Safe**: Pure functions, no side effects, declarative patterns
- ✅ **No Fallbacks**: Proper context flow, no brittle fallback logic
- ✅ **Single-File Role**: All timer logic in one file with clear separation

This is the **architecturally correct** solution that follows the established patterns and principles of the Universal Agent System.
