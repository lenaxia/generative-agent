# Next Session: Complete Memory Assessment System

## Session Context

You are continuing implementation of a dual-layer memory system with LLM-based importance assessment. **Phases 0-9 are complete** (26 commits, 131 tests passing). This session will complete **Phases 10-14** (estimated 4 hours).

## What's Already Working

### ✅ Unified Memory System (Phases 1-6)

**Files**:

- `common/providers/universal_memory_provider.py` - Provider with search
- `roles/shared_tools/memory_tools.py` - search_memory(), get_recent_memories()
- `common/intents.py` - MemoryWriteIntent
- `common/intent_processor.py` - Intent handler

**Capabilities**:

- Cross-role memory storage and retrieval
- 5 memory types: conversation, event, plan, preference, fact
- Importance-based TTL
- Rich metadata, tags, topics, summary
- Tiered loading (Tier 1: 5 recent)
- Sync reads (~50ms), async writes (~10ms)

**Tests**: 94 tests passing

### ✅ Universal Realtime Log (Phase 7)

**Files**:

- `common/realtime_log.py` - Realtime log utilities
- Updated: `roles/core_conversation.py`, `roles/core_calendar.py`, `roles/core_planning.py`

**Capabilities**:

- All roles write to realtime log
- Per-message TTL (24 hours)
- Analysis tracking (analyzed flags)
- Automatic cleanup
- Redis sorted set operations

**Tests**: 15 tests passing

### ✅ Memory Importance Assessment (Phases 8-9)

**Files**:

- `common/memory_assessment.py` - Pydantic model
- `common/memory_importance_assessor.py` - LLM assessor
- Updated: `common/providers/universal_memory_provider.py` (added summary, topics fields)

**Capabilities**:

- LLM-based assessment (WEAK model)
- Graduated TTL: >= 0.7 permanent, 0.5-0.7 = 30d, 0.3-0.5 = 7d, < 0.3 = 3d
- Summary generation
- Tag extraction (1-10)
- Topic identification (0-5)
- Structured JSON output with Pydantic validation

**Tests**: 22 tests passing

## Remaining Work (Phases 10-14)

### Phase 10: Analysis Integration (~2 hours)

**Goal**: Wire up the importance assessor to analyze realtime log and create assessed memories.

**Tasks**:

1. **Create analyze_conversation shared tool** (`roles/shared_tools/conversation_analysis.py`):

```python
@tool
def analyze_conversation() -> dict:
    """Trigger analysis of unanalyzed messages."""
    # Get unanalyzed messages from realtime log
    # Call MemoryImportanceAssessor
    # Create MemoryWriteIntent with assessment results
    # Mark messages as analyzed
```

2. **Create inactivity timeout checker** (`supervisor/scheduled_tasks.py` or new file):

```python
async def check_conversation_inactivity():
    """Check for conversations needing analysis (30min timeout)."""
    # Run every 5 minutes
    # Check last_message_time for each user
    # If > 30 min, trigger analysis
```

3. **Update role configurations**:

- Add `"conversation_analysis"` to shared tools for: conversation, calendar, planning
- NOT for: timer, weather, search (single-turn roles)

4. **Write tests** (~12 tests):

- analyze_conversation tool functionality
- Inactivity timeout triggers
- Assessment creates memory
- Graduated TTL applied
- Role configuration checks

### Phase 11: Dual-Layer Context Loading (~1 hour)

**Goal**: Load both realtime log and assessed memories in pre-processing.

**Tasks**:

1. **Update pre-processing functions**:

```python
def load_context(instruction, context, parameters):
    # Layer 1: Realtime log (last 10 messages)
    realtime_messages = get_recent_messages(user_id, limit=10)

    # Layer 2: Assessed memories (last 5, importance >= 0.7)
    assessed_memories = provider.get_recent_memories(
        user_id,
        min_importance=0.7,  # Only important
        limit=5
    )

    return {
        "realtime_context": realtime_messages,
        "assessed_memories": assessed_memories,
        "user_id": user_id
    }
```

2. **Update system prompts** (conversation, calendar, planning):

```
IMMEDIATE CONTEXT (last 10 turns):
{realtime_context}

IMPORTANT MEMORIES (assessed, permanent):
{assessed_memories}
```

3. **Write tests** (~9 tests):

- Dual-layer loading works
- Realtime context formatted correctly
- Assessed context with summaries
- Empty contexts handled

### Phase 12: Configuration (~30 mins)

**Goal**: Add memory system configuration.

**Tasks**:

1. **Update `config.yaml`**:

```yaml
memory_system:
  enabled: true

  # Importance thresholds
  permanent_threshold: 0.7

  # TTL calculation
  ttl_calculation:
    very_low_days: 3
    low_days: 7
    medium_days: 30

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

2. **Write tests** (~5 tests):

- Config loads correctly
- Defaults applied
- Thresholds used by assessor

### Phase 13: Integration Testing (~1 hour)

**Goal**: End-to-end tests for complete flow.

**File**: `tests/integration/test_memory_assessment_e2e.py`

**Tests** (~10 tests):

1. Multi-turn conversation → analysis → assessed memory
2. Inactivity timeout triggers analysis
3. Assessment creates memory with correct TTL
4. Dual-layer context loading works
5. Cross-role memory with realtime
6. Performance validation
7. Graduated TTL applied correctly
8. Summary and topics in memory
9. analyze_conversation tool works
10. Full workflow integration

### Phase 14: Final Documentation (~30 mins)

**Goal**: Update all documentation.

**Tasks**:

1. **Update `docs/57_UNIFIED_MEMORY_ARCHITECTURE.md`**:

- Add dual-layer architecture section
- Add importance assessment details
- Add graduated TTL table
- Add realtime log specification

2. **Update `README.md`**:

- Update test statistics (1000+ tests)
- Add memory assessment to features
- Update architecture highlights

3. **Update `docs/58_MEMORY_ASSESSMENT_IMPLEMENTATION_PLAN.md`**:

- Mark all phases complete
- Add final summary

## Implementation Rules

- ✅ Test-driven development: Write tests first
- ✅ Run tests after each change
- ✅ Commit after each phase
- ✅ No backwards compatibility code
- ✅ No legacy code
- ✅ Fix tech debt immediately
- ✅ Use grep to find all instances when refactoring

## Key Architecture

### Dual-Layer Memory

**Layer 1**: Realtime log (24h TTL, all roles, conversation flow)
**Layer 2**: Assessed memory (graduated TTL, important knowledge)

### Flow

```
User interaction
→ Saved to realtime log (24h TTL)
→ [30 min timeout OR analyze_conversation() call]
→ MemoryImportanceAssessor analyzes
→ Creates MemoryWriteIntent with assessment
→ Stored in unified memory with graduated TTL
→ Realtime messages marked as analyzed
→ Realtime expires after 24h
→ Assessed memory persists (3d to permanent)
```

### Graduated TTL

- importance >= 0.7: Permanent (no TTL)
- importance 0.5-0.7: 30 days
- importance 0.3-0.5: 7 days
- importance < 0.3: 3 days

### Context Loading

- Realtime: Last 10 messages (~200 tokens)
- Assessed: Last 5 important memories (~100 tokens)
- Total: ~300 tokens (10x reduction)

## Critical Files

**Core Implementation**:

- `common/realtime_log.py` - Realtime log utilities
- `common/memory_importance_assessor.py` - LLM assessor
- `common/memory_assessment.py` - Pydantic model
- `common/providers/universal_memory_provider.py` - Provider

**To Create**:

- `roles/shared_tools/conversation_analysis.py` - analyze tool
- `supervisor/scheduled_tasks.py` - Inactivity checker

**To Update**:

- Role pre-processing (dual-layer loading)
- Role system prompts
- `config.yaml` (memory_system section)

## Test Strategy

Each phase:

1. Create test file first
2. Implement functionality
3. Run tests, verify pass
4. Commit
5. Move to next phase

## Success Criteria

- ✅ All tests pass (100%)
- ✅ Realtime log integrated
- ✅ Importance assessment working
- ✅ Dual-layer context loading
- ✅ Configuration complete
- ✅ Documentation updated
- ✅ No tech debt

## Git Status

- **Branch**: main
- **Commits ahead**: 26 commits
- **Working tree**: Clean
- **Tests**: 131 unified memory tests passing

## Next Steps

Start with Phase 10.1: Create `roles/shared_tools/conversation_analysis.py` with the analyze_conversation tool. Follow the implementation plan in docs/58 and docs/59.

**The foundation is complete. Finish the integration to deliver a production-ready dual-layer memory system!**
</result>
</attempt_completion>
