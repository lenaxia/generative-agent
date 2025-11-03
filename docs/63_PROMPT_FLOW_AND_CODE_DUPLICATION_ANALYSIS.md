# Prompt Flow and Code Duplication Analysis

**Document ID:** 63
**Created:** 2025-11-03
**Status:** ANALYSIS
**Priority:** Medium
**Context:** Analysis of prompt consistency and code duplication in lifecycle functions

## Question 1: Prompt Consistency Throughout Flow

### Current Prompt Flow

```
User Input
  ↓
RequestMetadata.prompt (workflow_engine)
  ↓
UniversalAgent.execute_task(instruction)
  ↓
LLMSafeEventContext.original_prompt (NEW - added in this session)
  ↓
pre_data["_instruction"] (NEW - added in this session)
  ↓
Post-processing functions (conversation, calendar, planning)
  ↓
Realtime Log (user_message field)
```

### Consistency Analysis

**✅ CONSISTENT**: The prompt flows through consistently now with our fixes:

1. **Entry Point**: `RequestMetadata.prompt` contains user's original input
2. **Execution**: `instruction` parameter in `execute_task()`
3. **Context**: `event_context.original_prompt` set in universal_agent
4. **Pre-data**: `pre_data["_instruction"]` passed to post-processors
5. **Storage**: Captured in realtime log via post-processing

**Before Our Fix**: ❌ Broken at step 3-4 (context didn't have original_prompt)
**After Our Fix**: ✅ Complete flow from user input to storage

### Verification Points

```python
# 1. Workflow Engine creates RequestMetadata
RequestMetadata(prompt=instruction, ...)  # ✅

# 2. Universal Agent receives instruction
def execute_task(self, instruction: str, ...):  # ✅

# 3. Sets on event_context (OUR FIX)
event_context.original_prompt = instruction  # ✅

# 4. Passes via pre_data (OUR FIX)
pre_data_with_instruction = {**pre_data, "_instruction": instruction}  # ✅

# 5. Post-processors retrieve it (OUR FIX)
user_message = pre_data.get("_instruction", "unknown")  # ✅
```

**CONCLUSION**: ✅ Prompt is now handled consistently throughout the entire flow.

## Question 2: Code Duplication in Lifecycle Functions

### Current Duplication

#### Pre-processing Functions (3 roles)

**Conversation Role** (`load_conversation_context`):

```python
def load_conversation_context(instruction: str, context, parameters: dict) -> dict:
    from common.realtime_log import get_recent_messages
    from common.providers.universal_memory_provider import UniversalMemoryProvider

    user_id = getattr(context, "user_id", "unknown")
    realtime_messages = get_recent_messages(user_id, limit=10)

    memory_provider = UniversalMemoryProvider()
    assessed_memories = memory_provider.get_recent_memories(
        user_id=user_id, memory_types=["conversation", "event", "plan"], limit=5
    )

    important_memories = [m for m in assessed_memories if m.importance >= 0.7]

    return {
        "realtime_context": _format_realtime_messages(realtime_messages),
        "assessed_memories": _format_assessed_memories(important_memories),
        # ... role-specific fields
    }
```

**Calendar Role** (`load_calendar_context`): **~95% identical**
**Planning Role** (`load_planning_context`): **~95% identical**

**Duplicated Code**:

- Getting user_id from context
- Loading realtime messages
- Creating UniversalMemoryProvider
- Getting recent memories
- Filtering by importance
- Formatting functions

**Role-Specific Differences**:

- Memory types filter (conversation uses ["conversation", "event", "plan"], calendar uses ["event", "conversation", "plan"])
- Additional context fields (conversation has recent_topics, current_topics)

#### Post-processing Functions (3 roles)

**Conversation Role** (`save_message_to_log`):

```python
def save_message_to_log(llm_result: str, context, pre_data: dict) -> str:
    from common.realtime_log import add_message

    user_id = getattr(context, "user_id", "unknown")
    user_message = getattr(context, "original_prompt", None) or pre_data.get("_instruction", "unknown")

    add_message(
        user_id=user_id,
        user_message=user_message,
        assistant_response=llm_result,
        role="conversation",
        metadata=None,
    )
    return llm_result
```

**Calendar Role** (`save_calendar_event`): **~98% identical**
**Planning Role** (`save_planning_result`): **~98% identical**

**Duplicated Code**:

- Getting user_id from context
- Getting user_message from context/pre_data
- Calling add_message()
- Error handling
- Return llm_result

**Role-Specific Differences**:

- Role name ("conversation" vs "calendar" vs "planning")
- Planning converts llm_result to string

### Refactoring Options

#### Option 1: Shared Lifecycle Functions

**Approach**: Create shared functions in `roles/shared_tools/lifecycle_helpers.py`

```python
# roles/shared_tools/lifecycle_helpers.py

def load_dual_layer_context(
    context,
    memory_types: list[str] = None,
    additional_context: dict = None
) -> dict:
    """Shared pre-processing: Load dual-layer context."""
    from common.realtime_log import get_recent_messages
    from common.providers.universal_memory_provider import UniversalMemoryProvider

    user_id = getattr(context, "user_id", "unknown")

    # Layer 1: Realtime
    realtime_messages = get_recent_messages(user_id, limit=10)

    # Layer 2: Assessed
    provider = UniversalMemoryProvider()
    assessed_memories = provider.get_recent_memories(
        user_id=user_id,
        memory_types=memory_types or ["conversation", "event", "plan"],
        limit=5
    )

    important_memories = [m for m in assessed_memories if m.importance >= 0.7]

    base_context = {
        "realtime_context": _format_realtime_messages(realtime_messages),
        "assessed_memories": _format_assessed_memories(important_memories),
        "user_id": user_id,
    }

    # Merge with role-specific context
    if additional_context:
        base_context.update(additional_context)

    return base_context


def save_to_realtime_log(
    llm_result: str,
    context,
    pre_data: dict,
    role_name: str
) -> str:
    """Shared post-processing: Save to realtime log."""
    from common.realtime_log import add_message

    user_id = getattr(context, "user_id", "unknown")
    user_message = getattr(context, "original_prompt", None) or pre_data.get("_instruction", "unknown")

    add_message(
        user_id=user_id,
        user_message=user_message,
        assistant_response=str(llm_result),
        role=role_name,
        metadata=None,
    )

    return llm_result
```

**Usage in Roles**:

```python
# roles/core_conversation.py
def load_conversation_context(instruction: str, context, parameters: dict) -> dict:
    from roles.shared_tools.lifecycle_helpers import load_dual_layer_context

    # Load base context
    base = load_dual_layer_context(context, memory_types=["conversation", "event", "plan"])

    # Add conversation-specific context
    base["recent_topics"] = _load_recent_topics_cache(base["user_id"])
    base["current_topics"] = _extract_current_topics_simple(...)

    return base

def save_message_to_log(llm_result: str, context, pre_data: dict) -> str:
    from roles.shared_tools.lifecycle_helpers import save_to_realtime_log

    result = save_to_realtime_log(llm_result, context, pre_data, "conversation")

    # Additional conversation-specific logic
    _save_message_to_global_log(...)

    return result
```

**Pros**:

- ✅ Eliminates ~90% code duplication
- ✅ Single source of truth for dual-layer loading
- ✅ Easier to maintain and update
- ✅ Consistent behavior across roles
- ✅ Easier to test (test once, use everywhere)

**Cons**:

- ⚠️ Adds indirection (need to look in shared_tools)
- ⚠️ Roles become dependent on shared module
- ⚠️ Less obvious what each role does in isolation
- ⚠️ Requires refactoring existing tests

#### Option 2: Base Class/Mixin Pattern

**Approach**: Create a base class with common lifecycle methods

```python
# roles/base_role.py

class BaseMultiTurnRole:
    """Base class for multi-turn roles with dual-layer memory."""

    @staticmethod
    def load_dual_layer_context(context, memory_types=None):
        # Implementation
        pass

    @staticmethod
    def save_to_realtime_log(llm_result, context, pre_data, role_name):
        # Implementation
        pass
```

**Pros**:

- ✅ Clear inheritance hierarchy
- ✅ Type safety with base class
- ✅ Can override methods if needed

**Cons**:

- ❌ Violates single-file role pattern
- ❌ Adds complexity (inheritance)
- ❌ Not LLM-friendly (harder to understand)
- ❌ Goes against current architecture

#### Option 3: Keep Current Duplication

**Approach**: Accept the duplication as intentional

**Pros**:

- ✅ Each role is self-contained
- ✅ Easy to understand in isolation
- ✅ No dependencies between roles
- ✅ LLM-friendly (everything in one file)
- ✅ Follows single-file role pattern
- ✅ Easy to customize per role

**Cons**:

- ❌ Code duplication (~20 lines per role)
- ❌ Need to update 3 places for changes
- ❌ Risk of inconsistency

### Recommendation

**RECOMMENDED: Option 1 (Shared Lifecycle Functions)**

**Rationale**:

1. **Significant duplication**: ~95% identical code across 3 roles
2. **Low risk**: Shared functions are simple, well-defined
3. **High benefit**: Single source of truth for memory loading
4. **Maintainability**: Future changes only need 1 update
5. **Testability**: Test once, confidence everywhere
6. **Still LLM-friendly**: Shared functions are in `roles/shared_tools/` which is already a pattern

**Implementation Plan**:

1. Create `roles/shared_tools/lifecycle_helpers.py`
2. Move common code to `load_dual_layer_context()` and `save_to_realtime_log()`
3. Update 3 roles to use shared functions
4. Update tests to verify shared functions
5. Keep role-specific logic in roles (topics, etc.)

**Estimated Effort**: 1-2 hours
**Risk**: Low (well-tested, simple refactor)
**Benefit**: High (eliminates 60+ lines of duplication)

### Alternative: Hybrid Approach

Keep the **formatting functions** shared (already done):

- `_format_realtime_messages()` - duplicated 3x
- `_format_assessed_memories()` - duplicated 3x

Move these to `roles/shared_tools/memory_formatting.py`:

```python
def format_realtime_messages(messages):
    # Implementation
    pass

def format_assessed_memories(memories):
    # Implementation
    pass
```

This gives us **50% duplication reduction** with **minimal risk**.

## Conclusion

### Question 1: Prompt Consistency

✅ **YES** - Prompt is now handled consistently throughout the entire flow after our fixes.

### Question 2: Code Duplication

⚠️ **YES, SIGNIFICANT DUPLICATION EXISTS**

**Recommendation**: Implement Option 1 (Shared Lifecycle Functions) to eliminate 90% of duplication while maintaining LLM-friendly architecture.

**Priority**: Medium (system works correctly, but maintenance burden exists)

**Next Steps** (if pursuing refactor):

1. Create `roles/shared_tools/lifecycle_helpers.py`
2. Implement shared functions with tests
3. Refactor roles one at a time
4. Verify all tests still pass
5. Commit incrementally
