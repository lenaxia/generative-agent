# Legacy Code Removal Checklist

**Document ID:** 31
**Created:** 2025-10-15
**Status:** Analysis Complete - Ready for Implementation
**Priority:** High
**Context:** 100% LLM Development - Architecture Migration Cleanup

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

After systematically reviewing every Python file in the repository against the new LLM-optimized single-file role architecture and single event loop design from documents 25 and 26, I've identified significant amounts of legacy code that conflicts with the target architecture.

**Key Findings:**

- 25+ files in archived directory for complete removal
- 8 core system files requiring threading code removal
- ~2000+ lines of legacy code to be removed
- 60% reduction in role management complexity expected
- 100% alignment with LLM-optimized architecture achievable

## Major Categories of Legacy Code Identified

### 1. **ENTIRE ARCHIVED ROLES DIRECTORY** - Complete Removal Required

**Location:** `roles/archived/` (entire directory)
**Impact:** High Priority - Complete removal
**Rationale:** These multi-file roles conflict with the single-file role architecture from documents 25 & 26

**Files to Remove:**

- [ ] `roles/archived/analysis/definition.yaml`
- [ ] `roles/archived/calendar/definition.yaml`
- [ ] `roles/archived/calendar/lifecycle.py`
- [ ] `roles/archived/calendar/tools.py`
- [ ] `roles/archived/code_reviewer/definition.yaml`
- [ ] `roles/archived/code_reviewer/tools.py`
- [ ] `roles/archived/coding/definition.yaml`
- [ ] `roles/archived/default/definition.yaml`
- [ ] `roles/archived/no_tools/definition.yaml`
- [ ] `roles/archived/planning/definition.yaml`
- [ ] `roles/archived/planning/tools.py`
- [ ] `roles/archived/research_analyst/definition.yaml`
- [ ] `roles/archived/research_analyst/tools.py`
- [ ] `roles/archived/router/definition.yaml`
- [ ] `roles/archived/search/definition.yaml`
- [ ] `roles/archived/search/tools.py`
- [ ] `roles/archived/weather/definition.yaml`
- [ ] `roles/archived/weather/lifecycle.py`
- [ ] `roles/archived/weather/tools.py`
- [ ] `roles/archived/shared_tools/__init__.py`
- [ ] `roles/archived/shared_tools/data_processing.py`
- [ ] `roles/archived/shared_tools/event_subscriber.py`
- [ ] `roles/archived/shared_tools/file_operations.py`
- [ ] `roles/archived/shared_tools/redis_tools.py`
- [ ] `roles/archived/shared_tools/slack_tools.py`
- [ ] `roles/archived/shared_tools/summarizer_tools.py`

### 2. **THREADING-RELATED LEGACY CODE** - Conflicts with Single Event Loop

**Location:** `common/communication_manager.py`
**Rationale:** Threading conflicts with single event loop architecture from document 25

**Code to Remove:**

- [ ] Remove `import threading` (line 9)
- [ ] Remove `import queue` (line imports)
- [ ] Remove `_start_background_thread()` method (lines 115-136)
- [ ] Remove `_run_background_session()` method (lines 137-147)
- [ ] Remove `_background_session_loop()` method (lines 149-150)
- [ ] Remove `requires_background_thread()` method (lines 104-109)
- [ ] Remove threading setup in `__init__` methods
- [ ] Remove `self.background_thread` attribute
- [ ] Remove `self.message_queue` attribute
- [ ] Remove thread-related error handling

### 2B. **ADDITIONAL THREADING-RELATED LEGACY CODE** - Channel Handlers and Other Files

**Rationale:** Additional threading code discovered that conflicts with single event loop architecture

**Files with Threading Code to Remove:**

**Channel Handlers:**

- [ ] `common/channel_handlers/home_assistant_handler.py`: Remove `import threading` (line 12)
- [ ] `common/channel_handlers/home_assistant_handler.py`: Remove `_background_session_loop()` method (lines 120-122)
- [ ] `common/channel_handlers/voice_handler.py`: Remove `import threading` (line 10)
- [ ] `common/channel_handlers/voice_handler.py`: Remove `_background_session_loop()` method (lines 216-218)
- [ ] `common/channel_handlers/slack_handler.py`: Remove `import threading` (line 12)
- [ ] `common/channel_handlers/slack_handler.py`: Remove `_is_background_thread()` method (lines 215-218)
- [ ] `common/channel_handlers/slack_handler.py`: Remove `_background_session_loop()` method (lines 521-523)
- [ ] `common/channel_handlers/slack_handler.py`: Remove background thread detection logic (lines 332-334)

**Core Files:**

- [ ] `slack.py`: Remove `import threading` (line 11)
- [ ] `slack.py`: Remove `_process_ai_command_async()` background thread logic (lines 508-525)
- [ ] `slack.py`: Remove background thread processing in mention handlers (lines 683-714)
- [ ] `slack.py`: Remove background thread processing in DM handlers (lines 717-743)
- [ ] `roles/shared_tools/redis_tools.py`: Remove `import threading` (line 11)
- [ ] `roles/shared_tools/redis_tools.py`: Remove threading logic in async execution (lines 598-602)

**Test Files with Threading (Legacy Test Patterns):**

- [ ] `tests/integration/test_threading_fixes.py`: Remove `import threading` (line 14)
- [ ] `tests/integration/test_threading_fixes.py`: Remove background thread tests (lines 59-61, 288-290)
- [ ] `tests/unit/test_slack_workflow_integration_fix.py`: Remove `import threading` (line 11)
- [ ] `tests/unit/test_slack_workflow_integration_fix.py`: Remove threading test patterns (lines 65-68, 104-118)
- [ ] `tests/unit/test_resource_limits_unit.py`: Remove `import threading` (line 8)
- [ ] `tests/unit/test_resource_limits_unit.py`: Remove threading test logic (lines 234-236)
- [ ] `tests/unit/test_agent_pooling.py`: Remove threading test logic (lines 255-281)
- [ ] `tests/unit/test_slack_app_mention_fix.py`: Remove `import threading` (line 9)
- [ ] `tests/unit/test_slack_app_mention_fix.py`: Remove threading test patterns (lines 99-106)
- [ ] `tests/unit/test_bidirectional_communication.py`: Remove background thread requirement tests (lines 102-123)
- [ ] `tests/unit/test_unified_communication_manager.py`: Remove background thread tests (lines 240-250)

### 3. **BACKWARDS COMPATIBILITY CODE** - Role Registry

**Location:** `llm_provider/role_registry.py`
**Rationale:** Multi-file role support conflicts with single-file architecture

**Code to Remove:**

- [ ] Remove backward compatibility alias `self.roles = self.llm_roles` (lines 69-72)
- [ ] Remove multi-file role discovery code (lines 180-194)
- [ ] Remove `_load_multi_file_role()` method entirely (lines 283-326)
- [ ] Remove legacy tools.py loading (lines 305-310)
- [ ] Remove `_load_lifecycle_functions()` method (lines 665-671)
- [ ] Remove `_load_functions_from_file()` method (lines 673-702)
- [ ] Remove role type mapping compatibility code (lines 55-57)
- [ ] Remove lifecycle function registration for multi-file roles

### 4. **LEGACY MESSAGE BUS PATTERNS**

**Location:** `common/message_bus.py`
**Rationale:** Static patterns conflict with dynamic intent-based architecture

**Code to Remove:**

- [ ] Remove `import threading` (line 9)
- [ ] Remove backward compatibility comment (lines 30-32)
- [ ] Remove static MessageType enum (lines 24-50) - keep only dynamic registration
- [ ] Remove threading-related imports and setup

### 5. **SUPERVISOR THREADING REMNANTS**

**Location:** `supervisor/supervisor.py`
**Rationale:** Threading remnants conflict with single event loop

**Code to Remove:**

- [ ] Remove commented heartbeat references (lines 53-56)
- [ ] Remove commented import statement (line 23)
- [ ] Clean up `_scheduled_tasks` management (line 72)
- [ ] Remove heartbeat-related attributes

### 6. **CONFIGURATION LEGACY PATTERNS**

**Location:** `supervisor/supervisor_config.py`
**Rationale:** Threading-era configuration no longer needed

**Code to Remove:**

- [ ] Remove `heartbeat_interval: int = 30` (line 67)
- [ ] Remove `timer_check_interval: int = 5` (line 68)
- [ ] Remove `enable_heartbeat: bool = True` (line 111)
- [ ] Update configuration validation to reject these options

### 7. **TECH DEBT - UNUSED IMPORTS AND DEAD CODE**

**Multiple Locations:**

**Code to Remove:**

- [ ] `common/message_bus.py`: Remove unused `import threading`
- [ ] `common/communication_manager.py`: Remove unused threading imports
- [ ] Remove any remaining langchain compatibility test code
- [ ] Remove dead code comments throughout codebase
- [ ] Remove unused import statements

## Implementation Phases

### Phase 1: Complete Directory Removal (High Risk)

**Estimated Time:** 2 hours
**Risk Level:** High - Verify no hidden dependencies

- [ ] **Step 1.1:** Search entire codebase for references to archived roles
  ```bash
  grep -r "roles/archived" --include="*.py" .
  grep -r "archived/" --include="*.py" .
  ```
- [ ] **Step 1.2:** Remove entire `roles/archived/` directory
  ```bash
  rm -rf roles/archived/
  ```
- [ ] **Step 1.3:** Run full test suite to verify no breakage
- [ ] **Step 1.4:** Update any documentation references
- [ ] **Step 1.5:** Commit changes with descriptive message

### Phase 2: Threading Code Removal (High Risk)

**Estimated Time:** 6 hours
**Risk Level:** High - Ensure single event loop works

**Core Threading Removal:**

- [ ] **Step 2.1:** Remove threading imports from `common/communication_manager.py`
- [ ] **Step 2.2:** Remove `_start_background_thread()` method
- [ ] **Step 2.3:** Remove `_run_background_session()` method
- [ ] **Step 2.4:** Remove `_background_session_loop()` method
- [ ] **Step 2.5:** Remove `requires_background_thread()` method
- [ ] **Step 2.6:** Remove threading setup in `__init__` methods
- [ ] **Step 2.7:** Remove threading import from `common/message_bus.py`

**Channel Handler Threading Removal:**

- [ ] **Step 2.8:** Remove threading from all channel handlers:
  - `common/channel_handlers/home_assistant_handler.py`
  - `common/channel_handlers/voice_handler.py`
  - `common/channel_handlers/slack_handler.py`
- [ ] **Step 2.9:** Remove threading from `slack.py` main file
- [ ] **Step 2.10:** Remove threading from `roles/shared_tools/redis_tools.py`

**Testing and Validation:**

- [ ] **Step 2.11:** Test single event loop architecture thoroughly
- [ ] **Step 2.12:** Run threading validation tests
- [ ] **Step 2.13:** Verify all channel handlers work without threading
- [ ] **Step 2.14:** Commit changes

### Phase 3: Backwards Compatibility Removal (Medium Risk)

**Estimated Time:** 3 hours
**Risk Level:** Medium - Ensure all roles are single-file

- [ ] **Step 3.1:** Verify all active roles use single-file pattern
- [ ] **Step 3.2:** Remove `self.roles = self.llm_roles` alias
- [ ] **Step 3.3:** Remove multi-file role discovery code
- [ ] **Step 3.4:** Remove `_load_multi_file_role()` method entirely
- [ ] **Step 3.5:** Remove legacy lifecycle function loading
- [ ] **Step 3.6:** Remove role type mapping compatibility code
- [ ] **Step 3.7:** Test role registry functionality
- [ ] **Step 3.8:** Verify all single-file roles still work
- [ ] **Step 3.9:** Commit changes

### Phase 4: Configuration Cleanup (Low Risk)

**Estimated Time:** 1 hour
**Risk Level:** Low - Configuration changes

- [ ] **Step 4.1:** Remove `heartbeat_interval` from supervisor_config.py
- [ ] **Step 4.2:** Remove `timer_check_interval` from supervisor_config.py
- [ ] **Step 4.3:** Remove `enable_heartbeat` flag
- [ ] **Step 4.4:** Update configuration validation
- [ ] **Step 4.5:** Test configuration loading
- [ ] **Step 4.6:** Commit changes

### Phase 5: Message Bus Modernization (Low Risk)

**Estimated Time:** 2 hours
**Risk Level:** Low - Remove static patterns

- [ ] **Step 5.1:** Remove static MessageType enum
- [ ] **Step 5.2:** Remove backward compatibility comments
- [ ] **Step 5.3:** Ensure only dynamic message type registration is used
- [ ] **Step 5.4:** Test message bus functionality
- [ ] **Step 5.5:** Commit changes

### Phase 6: Test Cleanup (Medium Risk)

**Estimated Time:** 2 hours
**Risk Level:** Medium - Update test patterns

- [ ] **Step 6.1:** Remove threading-related test files:
  - `tests/integration/test_threading_fixes.py` (review if still needed)
  - Update threading tests to validate single event loop instead
- [ ] **Step 6.2:** Remove threading patterns from unit tests:
  - `tests/unit/test_slack_workflow_integration_fix.py`
  - `tests/unit/test_resource_limits_unit.py`
  - `tests/unit/test_agent_pooling.py`
  - `tests/unit/test_slack_app_mention_fix.py`
  - `tests/unit/test_bidirectional_communication.py`
  - `tests/unit/test_unified_communication_manager.py`
- [ ] **Step 6.3:** Update tests to validate single event loop architecture
- [ ] **Step 6.4:** Ensure test coverage is maintained
- [ ] **Step 6.5:** Commit test cleanup changes

### Phase 7: Final Cleanup (Low Risk)

**Estimated Time:** 1 hour
**Risk Level:** Low - Remove dead code

- [ ] **Step 7.1:** Remove unused imports throughout codebase
- [ ] **Step 7.2:** Remove dead code comments
- [ ] **Step 7.3:** Remove any remaining legacy compatibility code
- [ ] **Step 7.4:** Run final linting and formatting
- [ ] **Step 7.5:** Commit final cleanup

## Validation Steps

### Before Removal (Baseline)

- [ ] Run full test suite: `make test`
- [ ] Run linting: `make lint`
- [ ] Verify all roles are migrated to single-file pattern
- [ ] Confirm no production systems depend on legacy patterns
- [ ] Document current system state

### After Each Phase

- [ ] Run `make lint` to ensure code health
- [ ] Run full test suite: `make test`
- [ ] Test single event loop architecture
- [ ] Verify intent-based processing works
- [ ] Check for any broken imports or references

### Final Validation

- [ ] Confirm threading architecture compliance (only main thread)
- [ ] Verify all roles follow single-file pattern
- [ ] Test end-to-end workflows
- [ ] Performance benchmarking
- [ ] Integration testing
- [ ] Documentation updates

## Risk Mitigation

### High Risk Removals

1. **Entire archived roles directory**

   - Mitigation: Comprehensive grep search for dependencies
   - Rollback: Git revert if issues found

2. **Threading code in communication manager**
   - Mitigation: Thorough testing of single event loop
   - Rollback: Maintain feature branch until validation complete

### Medium Risk Removals

1. **Backwards compatibility in role registry**
   - Mitigation: Verify all roles are single-file before removal
   - Rollback: Staged removal with validation at each step

### Low Risk Removals

1. **Configuration and dead code**
   - Mitigation: Standard testing procedures
   - Rollback: Simple git revert if needed

## Success Metrics

### Quantitative Metrics

- [ ] **Files removed:** 25+ files in archived directory
- [ ] **Lines of code removed:** ~3000+ lines (increased due to additional threading code)
- [ ] **Files modified:** 15+ core system files (including channel handlers)
- [ ] **Threading imports removed:** 17+ files with threading imports
- [ ] **Test files updated:** 10+ test files with threading patterns
- [ ] **Test coverage maintained:** >90%
- [ ] **Performance improvement:** Measured reduction in complexity</search>
      </search_and_replace>

### Qualitative Metrics

- [ ] **Architecture compliance:** 100% alignment with documents 25 & 26
- [ ] **Code simplicity:** Significant reduction in role management complexity
- [ ] **Threading safety:** Single event loop architecture validated
- [ ] **LLM optimization:** Single-file role pattern fully implemented
- [ ] **Maintainability:** Cleaner, more focused codebase

## Post-Removal Tasks

### Documentation Updates

- [ ] Update architecture documentation to reflect changes
- [ ] Remove references to multi-file roles in guides
- [ ] Update deployment documentation
- [ ] Create migration notes for future reference

### Testing Enhancements

- [ ] Add tests to prevent regression to multi-file patterns
- [ ] Enhance threading architecture validation tests
- [ ] Add performance benchmarks for simplified architecture
- [ ] Create integration tests for single-file role system

### Monitoring and Validation

- [ ] Set up monitoring for threading violations
- [ ] Create alerts for legacy pattern usage
- [ ] Implement automated checks in CI/CD
- [ ] Regular architecture compliance audits

## Estimated Timeline

**Total Estimated Time:** 17 hours over 4-6 days
**Recommended Schedule:**

- Day 1: Phase 1 (Directory removal) + Phase 4 (Configuration) - 3 hours
- Day 2: Phase 2 (Threading removal) - High focus day - 6 hours
- Day 3: Phase 3 (Backwards compatibility) + Phase 5 (Message bus) - 5 hours
- Day 4: Phase 6 (Test cleanup) - 2 hours
- Day 5: Phase 7 (Final cleanup) + Validation - 1 hour
- Day 6: Documentation and post-removal tasks</search>
  </search_and_replace>

## Conclusion

This comprehensive removal will:

1. **Eliminate threading complexity** - Full compliance with single event loop architecture
2. **Simplify role management** - 60% reduction in complexity through single-file pattern
3. **Remove technical debt** - ~2000+ lines of legacy code removed
4. **Improve maintainability** - Cleaner, more focused codebase
5. **Enable LLM optimization** - Full alignment with documents 25 & 26 architecture

The removal is structured in phases with appropriate risk mitigation and validation steps to ensure a smooth transition to the new architecture while maintaining system stability and functionality.
