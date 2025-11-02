# Memory Importance Assessment Implementation Plan

**Document ID:** 58
**Created:** 2025-11-02
**Status:** IN PROGRESS - Phase 7.1 Complete (Realtime Log)
**Priority:** High
**Context:** Dual-layer memory with importance assessment

## Current Status

### ✅ Completed (Phase 7.1)

- Universal realtime log implementation
- Redis sorted set operations
- Per-message TTL (24 hours)
- Analysis tracking (analyzed flag)
- 9 tests passing

### 🚧 Remaining Work

#### Phase 7.2: Integrate Realtime Log into Roles

- Update conversation, calendar, planning post-processing
- Write to realtime log instead of/in addition to current systems
- 3 tests per role (9 tests total)

#### Phase 8: Memory Importance Assessor

- Create MemoryAssessment Pydantic model
- Implement MemoryImportanceAssessor class
- LLM-based assessment with BNF grammar
- Graduated TTL calculation
- ~15 tests

#### Phase 9: Schema Updates

- Add `summary` field to UniversalMemory
- Add `topics` field to UniversalMemory
- Update serialization
- Update provider TTL logic
- ~10 tests

#### Phase 10: Analysis Integration

- Create analyze_conversation shared tool
- Implement inactivity timeout checker
- Update role configurations
- ~12 tests

#### Phase 11: Dual-Layer Context Loading

- Update pre-processing to load both layers
- Update system prompts
- ~9 tests

#### Phase 12: Configuration

- Add memory_system config section
- Add importance thresholds
- Add TTL configuration
- ~5 tests

#### Phase 13: Integration Testing

- End-to-end assessment flow
- Dual-layer loading
- Performance validation
- ~10 tests

#### Phase 14: Documentation

- Update architecture docs
- Update README
- Add usage examples

**Estimated remaining time**: 10-12 hours

## Key Architecture Decisions

### 1. Dual-Layer Memory System

**Layer 1: Universal Realtime Log**

- Purpose: Track recent interactions for all roles
- Storage: `realtime:{user_id}` (Redis sorted set)
- TTL: 24 hours per message
- Access: All roles write, pre-processing reads
- Content: Full turn-by-turn exchanges

**Layer 2: Assessed Memory**

- Purpose: Long-term persistent knowledge
- Storage: `memory:{user_id}:{memory_id}` (Redis hash)
- TTL: Graduated based on importance
- Access: All roles via memory_tools
- Content: Consolidated, assessed, tagged

### 2. Importance-Based TTL

```python
if importance >= 0.7: return None  # Permanent
elif importance >= 0.5: return 30 * 24 * 3600  # 1 month
elif importance >= 0.3: return 7 * 24 * 3600   # 1 week
else: return 3 * 24 * 3600  # 3 days
```

### 3. Analysis Triggers

**Primary**: LLM calls `analyze_conversation()` tool
**Backup**: 30-minute inactivity timeout

### 4. Context Loading

```python
# Pre-processing loads both
{
    "realtime_context": get_recent_messages(limit=10),  # Immediate
    "assessed_memories": get_recent_memories(min_importance=0.7, limit=5)  # Long-term
}
```

## Implementation Pattern

### For Each Phase

1. **Write tests first** (TDD)
2. **Implement functionality**
3. **Run tests, verify they pass**
4. **Commit with descriptive message**
5. **Reflect on tech debt**
6. **Fix any issues immediately**

### Test Categories

- **Unit tests**: Component-level (mock Redis)
- **Integration tests**: End-to-end (real flow)
- **Performance tests**: Latency validation

### Commit Message Format

```
feat|fix|refactor|test|docs: brief description

- Bullet point 1
- Bullet point 2
- Test count and status
- Coverage percentage
```

## Next Steps

### Immediate (Phase 7.2)

**File**: `tests/unit/test_realtime_log_integration.py`

```python
def test_conversation_writes_to_realtime_log():
    """Test conversation role writes to realtime log."""
    from roles.core_conversation import save_message_to_log
    from common.realtime_log import get_recent_messages

    # Mock context
    context = MagicMock()
    context.user_id = "test_user"
    context.original_prompt = "Test message"

    # Execute
    save_message_to_log("Response", context, {})

    # Verify written to realtime log
    # (mock redis_zadd to verify)
```

**Implementation**: Update `save_message_to_log()` in conversation role to call `add_message()`.

### Critical Files to Modify

1. `roles/core_conversation.py` - Update post-processing
2. `roles/core_calendar.py` - Update post-processing
3. `roles/core_planning.py` - Update post-processing
4. `common/memory_assessment.py` - NEW: Pydantic models
5. `common/memory_importance_assessor.py` - NEW: LLM assessor
6. `roles/shared_tools/conversation_analysis.py` - NEW: analyze tool
7. `supervisor/scheduled_tasks.py` - NEW or UPDATE: inactivity checker
8. `config.yaml` - Add memory_system section

## BNF Grammar for Assessment

```bnf
<MemoryAssessment> ::= {
  "importance": <importance_score>,
  "summary": <summary_text>,
  "tags": [<tag>+],
  "topics": [<topic>*],
  "reasoning": <reasoning_text>
}

<importance_score> ::= <float_0_to_1>
<summary_text> ::= <string>
<tag> ::= <lowercase_string>
<topic> ::= <string>
<reasoning_text> ::= <string>
```

## Pydantic Model

```python
from pydantic import BaseModel, Field

class MemoryAssessment(BaseModel):
    importance: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(..., min_length=10)
    tags: list[str] = Field(..., min_items=1, max_items=10)
    topics: list[str] = Field(default_factory=list, max_items=5)
    reasoning: str = Field(..., max_length=500)

    class Config:
        extra = "forbid"
```

## Configuration Schema

```yaml
memory_system:
  enabled: true

  # Importance thresholds
  permanent_threshold: 0.7 # >= 0.7 = no TTL

  # TTL calculation
  ttl_calculation:
    very_low_days: 3 # importance < 0.3
    low_days: 7 # importance 0.3-0.5
    medium_days: 30 # importance 0.5-0.7

  # Realtime log
  realtime_log:
    ttl_hours: 24
    max_messages: 100
    cleanup_interval_minutes: 5

  # Analysis
  analysis:
    inactivity_timeout_minutes: 30
    model: "WEAK"
    timeout_seconds: 5
```

## Success Criteria

- ✅ All tests pass (100%)
- ✅ Realtime log working for all roles
- ✅ Importance assessment functional
- ✅ Graduated TTL implemented
- ✅ Dual-layer context loading working
- ✅ Inactivity timeout functional
- ✅ Configuration complete
- ✅ Documentation updated
- ✅ No tech debt remaining

## References

- [Document 57: Unified Memory Architecture](57_UNIFIED_MEMORY_ARCHITECTURE.md)
- [Document 21: High-Level Architecture Patterns](21_HIGH_LEVEL_ARCHITECTURE_PATTERNS.md)
- [Document 24: Unified Intent Processing](24_UNIFIED_INTENT_PROCESSING_ARCHITECTURE.md)
