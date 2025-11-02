# Memory System Implementation Plan - Phases 10-14

**Document ID:** 61
**Created:** 2025-11-02
**Status:** READY FOR IMPLEMENTATION
**Priority:** High
**Context:** Complete remaining phases of dual-layer memory system

## Executive Summary

This document provides a detailed, step-by-step implementation plan for completing Phases 10-14 of the memory system. The foundation (Phases 0-9) is complete with 131 tests passing. This plan focuses on integration, configuration, testing, and documentation.

**Estimated Time**: 6-8 hours
**Estimated Tests**: 36 new tests
**Commits**: 5-7 commits

## Prerequisites

### ✅ Already Complete (Phases 0-9)

- Universal memory system (94 tests)
- Universal realtime log (15 tests)
- Memory importance assessment (22 tests)
- Core infrastructure fully functional

### 📋 Files Already Implemented

- `common/providers/universal_memory_provider.py`
- `common/memory_assessment.py`
- `common/memory_importance_assessor.py`
- `common/realtime_log.py`
- `roles/shared_tools/memory_tools.py`
- `common/intents.py` (MemoryWriteIntent)

## Phase 10: Analysis Integration (2-3 hours)

### Goal

Wire up the importance assessor to analyze realtime log and create assessed memories.

### 10.1: Create Conversation Analysis Tool (45 mins)

**File**: `roles/shared_tools/conversation_analysis.py` (NEW)

**Implementation**:

```python
"""Conversation analysis tool for triggering memory importance assessment.

This tool allows roles to trigger analysis of unanalyzed messages in the
realtime log, creating assessed memories with importance scores and metadata.
"""

import logging
from typing import Any

from strands import tool

from common.memory_assessment import MemoryAssessment
from common.memory_importance_assessor import MemoryImportanceAssessor
from common.realtime_log import get_unanalyzed_messages, mark_as_analyzed
from common.intents import MemoryWriteIntent
from llm_provider.factory import LLMFactory

logger = logging.getLogger(__name__)


@tool
def analyze_conversation() -> dict[str, Any]:
    """Trigger analysis of unanalyzed messages in realtime log.

    This tool analyzes recent unanalyzed messages using an LLM to assess
    their importance and generate structured metadata. Assessed memories
    are stored with graduated TTL based on importance.

    Returns:
        Dict with analysis results including count of memories created
    """
    # Implementation details in test file
    pass
```

**Test File**: `tests/unit/test_conversation_analysis.py` (NEW)

**Tests** (6 tests):

1. `test_analyze_conversation_success()` - Successful analysis creates memories
2. `test_analyze_conversation_no_unanalyzed()` - Handles no unanalyzed messages
3. `test_analyze_conversation_marks_analyzed()` - Messages marked as analyzed
4. `test_analyze_conversation_graduated_ttl()` - Correct TTL applied
5. `test_analyze_conversation_assessment_failure()` - Handles assessment failures gracefully
6. `test_analyze_conversation_multiple_messages()` - Analyzes multiple messages

**Implementation Steps**:

1. Write tests first (TDD)
2. Implement `analyze_conversation()` function:
   - Get unanalyzed messages from realtime log
   - Initialize MemoryImportanceAssessor
   - For each message, call `assess_memory()`
   - Create MemoryWriteIntent with assessment results
   - Mark messages as analyzed
   - Return summary
3. Run tests with timeout: `pytest tests/unit/test_conversation_analysis.py -v --timeout=30`
4. Commit: `feat: add conversation analysis tool for memory assessment`

### 10.2: Create Inactivity Timeout Checker (45 mins)

**File**: `supervisor/scheduled_tasks.py` (NEW or UPDATE if exists)

**Implementation**:

```python
"""Scheduled tasks for background processing.

This module provides scheduled tasks that run periodically to handle
background operations like conversation inactivity checking.
"""

import asyncio
import logging
import time
from typing import Any

from common.realtime_log import get_last_message_time, get_unanalyzed_messages
from roles.shared_tools.conversation_analysis import analyze_conversation

logger = logging.getLogger(__name__)


async def check_conversation_inactivity(
    inactivity_timeout_minutes: int = 30
) -> dict[str, Any]:
    """Check for conversations needing analysis due to inactivity.

    Runs periodically to check if users have unanalyzed messages and
    their last message was more than the timeout period ago.

    Args:
        inactivity_timeout_minutes: Minutes of inactivity before triggering analysis

    Returns:
        Dict with analysis results for each user
    """
    # Implementation details in test file
    pass
```

**Test File**: `tests/unit/test_scheduled_tasks.py` (NEW)

**Tests** (6 tests):

1. `test_inactivity_checker_triggers_analysis()` - Analysis triggered after timeout
2. `test_inactivity_checker_no_trigger_recent()` - No trigger for recent activity
3. `test_inactivity_checker_multiple_users()` - Handles multiple users
4. `test_inactivity_checker_no_unanalyzed()` - Skips users with no unanalyzed messages
5. `test_inactivity_checker_timeout_configuration()` - Respects timeout config
6. `test_inactivity_checker_error_handling()` - Handles errors gracefully

**Implementation Steps**:

1. Write tests first
2. Implement `check_conversation_inactivity()`:
   - Get list of users with unanalyzed messages
   - For each user, check last message time
   - If > timeout, trigger analysis
   - Log results
3. Run tests: `pytest tests/unit/test_scheduled_tasks.py -v --timeout=30`
4. Commit: `feat: add inactivity timeout checker for automatic analysis`

### 10.3: Update Role Configurations (30 mins)

**Files to Update**:

- `roles/core_conversation.py`
- `roles/core_calendar.py`
- `roles/core_planning.py`

**Changes**:

Add `"conversation_analysis"` to shared tools:

```python
"tools": {
    "shared": ["redis_tools", "memory_tools", "conversation_analysis"],
}
```

**Test File**: `tests/unit/test_role_analysis_integration.py` (NEW)

**Tests** (3 tests):

1. `test_conversation_role_has_analysis_tool()` - Conversation role includes tool
2. `test_calendar_role_has_analysis_tool()` - Calendar role includes tool
3. `test_planning_role_has_analysis_tool()` - Planning role includes tool

**Implementation Steps**:

1. Write tests first
2. Update each role's ROLE_CONFIG
3. Run tests: `pytest tests/unit/test_role_analysis_integration.py -v`
4. Commit: `feat: integrate conversation analysis tool into multi-turn roles`

## Phase 11: Dual-Layer Context Loading (1-1.5 hours)

### Goal

Load both realtime log and assessed memories in pre-processing.

### 11.1: Update Pre-processing Functions (45 mins)

**Files to Update**:

- `roles/core_conversation.py` - Update `load_conversation_context()`
- `roles/core_calendar.py` - Update `load_context()`
- `roles/core_planning.py` - Update `load_context()`

**Implementation Pattern**:

```python
def load_conversation_context(instruction, context, parameters):
    """Load dual-layer context: realtime log + assessed memories."""
    from common.realtime_log import get_recent_messages
    from common.providers.universal_memory_provider import UniversalMemoryProvider

    user_id = context.user_id

    # Layer 1: Realtime log (last 10 messages)
    realtime_messages = get_recent_messages(user_id, limit=10)

    # Layer 2: Assessed memories (last 5, importance >= 0.7)
    provider = UniversalMemoryProvider()
    assessed_memories = provider.get_recent_memories(
        user_id=user_id,
        memory_types=["conversation", "event", "plan"],
        limit=5
    )

    # Filter for important memories only
    important_memories = [
        m for m in assessed_memories
        if m.get("importance", 0) >= 0.7
    ]

    return {
        "realtime_context": _format_realtime_messages(realtime_messages),
        "assessed_memories": _format_assessed_memories(important_memories),
        "message_count": len(realtime_messages),
        "unanalyzed_count": sum(1 for m in realtime_messages if not m.get("analyzed")),
        "user_id": user_id
    }

def _format_realtime_messages(messages):
    """Format realtime messages for prompt."""
    if not messages:
        return "No recent messages."

    formatted = []
    for msg in messages:
        formatted.append(f"User: {msg['user']}")
        formatted.append(f"Assistant: {msg['assistant']}")
    return "\n".join(formatted)

def _format_assessed_memories(memories):
    """Format assessed memories for prompt."""
    if not memories:
        return "No important memories."

    formatted = []
    for mem in memories:
        summary = mem.get("summary", mem.get("content", ""))
        tags = ", ".join(mem.get("tags", []))
        formatted.append(f"- {summary} (tags: {tags})")
    return "\n".join(formatted)
```

**Test File**: `tests/unit/test_dual_layer_loading.py` (NEW)

**Tests** (9 tests):

1. `test_conversation_loads_both_layers()` - Both layers loaded
2. `test_calendar_loads_both_layers()` - Both layers loaded
3. `test_planning_loads_both_layers()` - Both layers loaded
4. `test_realtime_context_formatted()` - Realtime messages formatted correctly
5. `test_assessed_context_formatted()` - Assessed memories formatted correctly
6. `test_empty_realtime_handled()` - Empty realtime log handled
7. `test_empty_assessed_handled()` - Empty assessed memories handled
8. `test_importance_filtering()` - Only >= 0.7 importance loaded
9. `test_message_counts_accurate()` - Counts in context are accurate

**Implementation Steps**:

1. Write tests first
2. Update pre-processing functions in each role
3. Add helper formatting functions
4. Run tests: `pytest tests/unit/test_dual_layer_loading.py -v --timeout=30`
5. Commit: `feat: implement dual-layer context loading in roles`

### 11.2: Update System Prompts (15 mins)

**Files to Update**:

- `roles/core_conversation.py`
- `roles/core_calendar.py`
- `roles/core_planning.py`

**Changes**:

Update system prompts to include both layers:

```python
"system": """You are a conversational AI assistant.

IMMEDIATE CONTEXT (last 10 turns):
{{realtime_context}}

IMPORTANT MEMORIES (assessed, permanent):
{{assessed_memories}}

Message count: {{message_count}}
Unanalyzed: {{unanalyzed_count}}

Use both immediate context and important memories to maintain continuity.
"""
```

**No new tests needed** - covered by existing role tests

**Implementation Steps**:

1. Update system prompts in ROLE_CONFIG
2. Run existing role tests to verify: `pytest tests/unit/test_*_role.py -v`
3. Commit: `feat: update role prompts for dual-layer memory context`

## Phase 12: Configuration (30-45 mins)

### Goal

Add memory system configuration to config.yaml.

### 12.1: Update Configuration File (15 mins)

**File**: `config.yaml`

**Changes**:

Add new section:

```yaml
# Memory System Configuration
memory_system:
  enabled: true

  # Importance thresholds
  permanent_threshold: 0.7 # >= 0.7 = no TTL

  # TTL calculation (days)
  ttl_calculation:
    very_low_days: 3 # importance < 0.3
    low_days: 7 # importance 0.3-0.5
    medium_days: 30 # importance 0.5-0.7

  # Realtime log settings
  realtime_log:
    ttl_hours: 24
    max_messages: 100
    cleanup_interval_minutes: 5

  # Analysis settings
  analysis:
    inactivity_timeout_minutes: 30
    model: "WEAK"
    timeout_seconds: 5
```

**Implementation Steps**:

1. Add configuration section to config.yaml
2. Commit: `feat: add memory system configuration`

### 12.2: Create Configuration Tests (30 mins)

**Test File**: `tests/unit/test_memory_config.py` (NEW)

**Tests** (5 tests):

1. `test_config_loads_correctly()` - Config loads without errors
2. `test_default_values_applied()` - Defaults used when not specified
3. `test_thresholds_used_by_assessor()` - Assessor uses config thresholds
4. `test_ttl_calculation_uses_config()` - TTL calculation uses config
5. `test_inactivity_timeout_configurable()` - Timeout checker uses config

**Implementation Steps**:

1. Write tests first
2. Update MemoryImportanceAssessor to read from config
3. Update scheduled tasks to read from config
4. Run tests: `pytest tests/unit/test_memory_config.py -v`
5. Commit: `feat: integrate memory system configuration`

## Phase 13: Integration Testing (1-1.5 hours)

### Goal

End-to-end tests for complete flow.

### 13.1: Create Integration Test Suite (1-1.5 hours)

**File**: `tests/integration/test_memory_assessment_e2e.py` (NEW)

**Tests** (10 tests):

1. `test_multi_turn_conversation_to_assessed_memory()` - Full flow
2. `test_inactivity_timeout_triggers_analysis()` - Timeout works
3. `test_assessment_creates_memory_with_correct_ttl()` - TTL applied
4. `test_dual_layer_context_loading_works()` - Both layers load
5. `test_cross_role_memory_with_realtime()` - Cross-role integration
6. `test_performance_validation()` - Latency within targets
7. `test_graduated_ttl_applied_correctly()` - All TTL tiers work
8. `test_summary_and_topics_in_memory()` - Metadata present
9. `test_analyze_conversation_tool_works()` - Tool callable
10. `test_full_workflow_integration()` - Complete workflow

**Implementation Steps**:

1. Write all tests first
2. Run tests: `pytest tests/integration/test_memory_assessment_e2e.py -v --timeout=60`
3. Fix any integration issues discovered
4. Commit: `test: add comprehensive memory assessment integration tests`

## Phase 14: Final Documentation (30-45 mins)

### Goal

Update all documentation to reflect completed implementation.

### 14.1: Update Architecture Documentation (15 mins)

**File**: `docs/57_UNIFIED_MEMORY_ARCHITECTURE.md`

**Changes**:

1. Update status to "COMPLETE"
2. Add dual-layer architecture section
3. Add importance assessment details
4. Add graduated TTL table
5. Add realtime log specification
6. Update test counts

**Implementation Steps**:

1. Update document
2. Commit: `docs: update memory architecture with dual-layer details`

### 14.2: Update README (10 mins)

**File**: `README.md`

**Changes**:

1. Update test statistics (160+ tests)
2. Add memory assessment to features
3. Update architecture highlights

**Implementation Steps**:

1. Update README
2. Commit: `docs: update README with memory system features`

### 14.3: Update Implementation Plan (10 mins)

**File**: `docs/58_MEMORY_ASSESSMENT_IMPLEMENTATION_PLAN.md`

**Changes**:

1. Mark all phases complete
2. Add final summary
3. Add lessons learned

**Implementation Steps**:

1. Update document
2. Commit: `docs: mark memory assessment implementation complete`

## Implementation Checklist

### Phase 10: Analysis Integration

- [ ] 10.1: Create conversation_analysis.py (6 tests)
- [ ] 10.2: Create scheduled_tasks.py (6 tests)
- [ ] 10.3: Update role configurations (3 tests)
- [ ] Commit 1: conversation analysis tool
- [ ] Commit 2: inactivity timeout checker
- [ ] Commit 3: role integration

### Phase 11: Dual-Layer Context Loading

- [ ] 11.1: Update pre-processing functions (9 tests)
- [ ] 11.2: Update system prompts (0 new tests)
- [ ] Commit 4: dual-layer loading
- [ ] Commit 5: updated prompts

### Phase 12: Configuration

- [ ] 12.1: Update config.yaml
- [ ] 12.2: Create configuration tests (5 tests)
- [ ] Commit 6: configuration

### Phase 13: Integration Testing

- [ ] 13.1: Create integration test suite (10 tests)
- [ ] Commit 7: integration tests

### Phase 14: Documentation

- [ ] 14.1: Update architecture docs
- [ ] 14.2: Update README
- [ ] 14.3: Update implementation plan
- [ ] Commit 8: documentation updates

## Success Criteria

- [ ] All 36 new tests pass (100%)
- [ ] Total test count: 167+ tests
- [ ] Realtime log integrated into all multi-turn roles
- [ ] Importance assessment working end-to-end
- [ ] Dual-layer context loading functional
- [ ] Configuration complete and tested
- [ ] Documentation updated
- [ ] No tech debt remaining
- [ ] Performance targets met:
  - [ ] Tier 1 load: < 50ms
  - [ ] Assessment: < 300ms
  - [ ] Memory write: < 20ms

## Testing Strategy

### Test Execution Order

1. **Unit tests first**: Verify individual components
2. **Integration tests second**: Verify end-to-end flow
3. **Performance tests last**: Validate latency targets

### Test Commands

```bash
# Run all new unit tests
pytest tests/unit/test_conversation_analysis.py \
       tests/unit/test_scheduled_tasks.py \
       tests/unit/test_role_analysis_integration.py \
       tests/unit/test_dual_layer_loading.py \
       tests/unit/test_memory_config.py \
       -v --timeout=30

# Run integration tests
pytest tests/integration/test_memory_assessment_e2e.py -v --timeout=60

# Run full test suite
pytest tests/ -v --timeout=60

# Check coverage
pytest tests/ --cov=common --cov=roles --cov=supervisor --cov-report=html
```

## Risk Mitigation

### Potential Issues

1. **LLM timeout**: Assessor may timeout on slow responses

   - **Mitigation**: 5-second timeout with graceful fallback

2. **Redis connection**: Redis may be unavailable

   - **Mitigation**: Existing error handling in realtime_log.py

3. **Memory overhead**: Dual-layer loading may increase prompt size

   - **Mitigation**: Limit to 10 realtime + 5 assessed messages

4. **Analysis frequency**: Too frequent analysis may impact performance
   - **Mitigation**: 30-minute inactivity timeout + LLM-triggered only

### Rollback Plan

If critical issues arise:

1. Disable memory system via config: `memory_system.enabled: false`
2. Roles fall back to existing memory_tools only
3. No data loss - realtime log and assessed memories preserved
4. Can re-enable after fixes

## Performance Targets

| Operation    | Target   | Measurement            |
| ------------ | -------- | ---------------------- |
| Tier 1 load  | < 50ms   | Pre-processing latency |
| Tier 2 load  | < 50ms   | Pre-processing latency |
| Assessment   | < 300ms  | LLM call duration      |
| Memory write | < 20ms   | Async write latency    |
| Full request | < 1500ms | End-to-end with LLM    |

## Next Steps

1. **Start with Phase 10.1**: Create conversation analysis tool
2. **Follow TDD**: Write tests first, then implement
3. **Commit frequently**: After each sub-phase
4. **Run tests continuously**: Verify no regressions
5. **Update this document**: Check off completed items

## References

- [Doc 57: Unified Memory Architecture](57_UNIFIED_MEMORY_ARCHITECTURE.md)
- [Doc 58: Memory Assessment Implementation Plan](58_MEMORY_ASSESSMENT_IMPLEMENTATION_PLAN.md)
- [Doc 59: Memory Importance Assessor Design](59_MEMORY_IMPORTANCE_ASSESSOR_DESIGN.md)
- [Doc 60: Next Session Memory Implementation](60_NEXT_SESSION_MEMORY_IMPLEMENTATION.md)

---

**Ready to implement**: This plan provides step-by-step instructions for completing the memory system. Follow the phases sequentially, write tests first, and commit after each sub-phase.
