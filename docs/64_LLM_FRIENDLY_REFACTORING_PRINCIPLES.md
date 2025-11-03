# LLM-Friendly Refactoring Principles

**Document ID:** 64
**Created:** 2025-11-03
**Status:** GUIDELINES
**Priority:** High
**Context:** Maintaining LLM-friendliness while reducing code duplication

## Core Principle

**LLM-Friendly Code** = Code that an LLM can easily understand, modify, and reason about without extensive context.

## Key Characteristics of LLM-Friendly Code

### 1. Locality of Behavior

**Principle**: Related code should be physically close together

**Good** (LLM-friendly):

```python
# roles/core_conversation.py - Everything in one file
def load_context():
    # Load memory
    pass

def save_to_log():
    # Save memory
    pass
```

**Bad** (Not LLM-friendly):

```python
# roles/core_conversation.py
from roles.base import BaseRole

class ConversationRole(BaseRole):
    # Behavior split across multiple files
    pass
```

### 2. Explicit Over Implicit

**Principle**: Make behavior obvious, not hidden

**Good**:

```python
def load_context():
    from roles.shared_tools.lifecycle_helpers import load_dual_layer_context
    # ✅ Explicit import shows where function comes from
    return load_dual_layer_context(context, memory_types=["conversation"])
```

**Bad**:

```python
class ConversationRole(BaseRole):
    # ❌ Implicit: Where does load_context come from?
    pass
```

### 3. Flat Over Nested

**Principle**: Minimize inheritance and nesting

**Good**:

```python
# Flat function calls
result = load_dual_layer_context(...)
```

**Bad**:

```python
# Deep inheritance
class ConversationRole(BaseMultiTurnRole(MemoryEnabledRole(BaseRole))):
    pass
```

### 4. Self-Documenting Code

**Principle**: Code should explain itself

**Good**:

```python
def load_dual_layer_context(
    context,
    memory_types: list[str] = None,
    realtime_limit: int = 10,
    assessed_limit: int = 5,
    importance_threshold: float = 0.7
) -> dict:
    """Load both realtime log and assessed memories.

    Args:
        context: Event context with user_id
        memory_types: Types to filter (default: all)
        realtime_limit: Max realtime messages (default: 10)
        assessed_limit: Max assessed memories (default: 5)
        importance_threshold: Min importance (default: 0.7)

    Returns:
        Dict with realtime_context and assessed_memories
    """
```

**Bad**:

```python
def load_ctx(c, mt=None):  # ❌ Unclear names, no docs
    pass
```

### 5. Minimal Abstraction Layers

**Principle**: Don't abstract too early

**Good**:

```python
# One level of abstraction
def load_dual_layer_context():
    realtime = get_recent_messages()
    assessed = get_recent_memories()
    return format_context(realtime, assessed)
```

**Bad**:

```python
# Too many layers
class ContextLoader:
    def __init__(self):
        self.strategy = DualLayerStrategy()

    def load(self):
        return self.strategy.execute(self.get_provider())
```

## Applying to Our Refactoring

### Proposed Refactoring: LLM-Friendliness Analysis

#### ✅ Maintains LLM-Friendliness

**1. Explicit Imports**

```python
# roles/core_conversation.py
def load_conversation_context(instruction: str, context, parameters: dict) -> dict:
    from roles.shared_tools.lifecycle_helpers import load_dual_layer_context
    # ✅ LLM can see exactly where function comes from
```

**2. Clear Function Names**

```python
load_dual_layer_context()  # ✅ Self-explanatory
save_to_realtime_log()     # ✅ Clear purpose
```

**3. Minimal Abstraction**

```python
# Only one level: role → shared function → implementation
# No inheritance, no complex patterns
```

**4. Comprehensive Documentation**

```python
def load_dual_layer_context(...):
    """Load both realtime log and assessed memories.

    This is the standard pre-processing function for multi-turn roles.
    It loads:
    1. Layer 1: Realtime log (last N messages, 24h TTL)
    2. Layer 2: Assessed memories (important memories, graduated TTL)

    Used by: conversation, calendar, planning roles
    """
```

**5. Still in Single File Context**

```python
# roles/core_conversation.py
# LLM can still see:
# - Role configuration
# - Tools
# - Lifecycle functions (now just calls)
# - Intent handlers
# All in one file!
```

#### ⚠️ Potential LLM-Friendliness Concerns

**Concern 1**: "LLM needs to look in two files"
**Mitigation**:

- Shared function is in `roles/shared_tools/` (established pattern)
- Import is explicit at call site
- Function name is descriptive
- LLM can easily find and read shared function

**Concern 2**: "Less obvious what role does"
**Mitigation**:

- Role file still shows the call with parameters
- Parameters make behavior explicit
- Docstring in role explains what happens
- Example:

```python
def load_conversation_context(...):
    """Load dual-layer context for conversation role.

    Loads realtime log and assessed memories using shared helper.
    Adds conversation-specific context (topics, etc.).
    """
    base = load_dual_layer_context(
        context,
        memory_types=["conversation", "event", "plan"],  # ✅ Explicit
        realtime_limit=10,  # ✅ Clear
        assessed_limit=5,   # ✅ Obvious
    )
    # Add conversation-specific context
    base["recent_topics"] = ...
    return base
```

**Concern 3**: "Harder to modify individual roles"
**Mitigation**:

- Roles can still override behavior
- Shared function has parameters for customization
- Can add role-specific logic before/after shared call
- Example:

```python
def load_conversation_context(...):
    # Pre-shared logic
    if special_case:
        return custom_context()

    # Shared logic
    base = load_dual_layer_context(...)

    # Post-shared logic
    base["custom_field"] = ...
    return base
```

## LLM-Friendly Refactoring Guidelines

### DO ✅

1. **Use Explicit Imports**

   ```python
   from roles.shared_tools.lifecycle_helpers import load_dual_layer_context
   ```

2. **Keep Function Calls in Role Files**

   ```python
   # Role file shows what happens
   def load_context(...):
       return load_dual_layer_context(context, memory_types=["conversation"])
   ```

3. **Use Descriptive Names**

   ```python
   load_dual_layer_context()  # Not: load_ctx()
   save_to_realtime_log()     # Not: save_log()
   ```

4. **Document Shared Functions Thoroughly**

   ```python
   def load_dual_layer_context(...):
       """Complete docstring with:
       - What it does
       - Why it exists
       - Who uses it
       - Parameters explained
       - Return value explained
       """
   ```

5. **Keep Shared Functions Simple**
   ```python
   # One clear purpose, no complex logic
   def load_dual_layer_context():
       # Load realtime
       # Load assessed
       # Format
       # Return
   ```

### DON'T ❌

1. **Don't Use Inheritance**

   ```python
   # ❌ LLM has to understand class hierarchy
   class ConversationRole(BaseMultiTurnRole):
       pass
   ```

2. **Don't Hide Behavior**

   ```python
   # ❌ Magic happens somewhere
   @auto_load_memory
   def my_function():
       pass
   ```

3. **Don't Create Deep Abstractions**

   ```python
   # ❌ Too many layers
   ContextLoaderFactory.create().get_strategy().load()
   ```

4. **Don't Use Complex Patterns**

   ```python
   # ❌ LLM has to understand pattern
   class MemoryLoadingStrategy(ABC):
       @abstractmethod
       def load(self): pass
   ```

5. **Don't Split Related Code**
   ```python
   # ❌ Behavior scattered
   # roles/core_conversation.py
   # roles/mixins/memory_mixin.py
   # roles/base/base_role.py
   ```

## Proposed Refactoring: LLM-Friendliness Score

### Before Refactoring

- **Locality**: ✅ 10/10 (everything in role file)
- **Explicitness**: ✅ 10/10 (all code visible)
- **Flatness**: ✅ 10/10 (no inheritance)
- **Self-Documentation**: ✅ 8/10 (good docs)
- **Abstraction**: ✅ 10/10 (minimal)
- **Overall**: ✅ 9.6/10

### After Refactoring (Shared Functions)

- **Locality**: ✅ 8/10 (one hop to shared_tools)
- **Explicitness**: ✅ 10/10 (explicit imports)
- **Flatness**: ✅ 10/10 (still no inheritance)
- **Self-Documentation**: ✅ 10/10 (better docs)
- **Abstraction**: ✅ 9/10 (one simple abstraction)
- **Overall**: ✅ 9.4/10

**Verdict**: ✅ **Minimal LLM-friendliness impact** (-0.2 points)
**Benefit**: 🎯 **Eliminates 165 lines of duplication**

## Recommendation

✅ **PROCEED with refactoring using shared functions**

**Rationale**:

1. Maintains 94% LLM-friendliness score
2. Eliminates 95-98% code duplication
3. Follows established `shared_tools` pattern
4. Simple, flat, explicit design
5. Well-documented with clear purpose
6. Easy for LLM to understand and modify

**Implementation Approach**:

- Create `roles/shared_tools/lifecycle_helpers.py`
- Keep functions simple and well-documented
- Use explicit imports in roles
- Maintain role-specific customization ability
- Test thoroughly

**LLM-Friendly Checklist for Implementation**:

- ✅ Explicit imports at call site
- ✅ Descriptive function names
- ✅ Comprehensive docstrings
- ✅ Simple, flat function calls
- ✅ No inheritance or complex patterns
- ✅ Clear parameter names
- ✅ Obvious return values
- ✅ Role files still show what happens

The refactoring maintains LLM-friendliness while significantly improving maintainability.
