# Lifecycle Hook System Implementation - Handoff Document

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

## **CURRENT STATUS: PHASE 1 COMPLETE ✅**

The lifecycle hook system has been **partially implemented** with major architectural improvements completed. The system is currently at commit `95a7579` which represents a stable foundation.

## **WHAT HAS BEEN ACCOMPLISHED ✅**

### **Phase 1: Architecture Cleanup & Foundation (COMPLETE)**

- ✅ **Removed ExecutionMode enum** and all execution type complexity
- ✅ **Simplified RoleRegistry** - removed `get_role_execution_type()` method
- ✅ **Updated UniversalAgent** to use unified execution for all roles
- ✅ **Removed weather special case** from WorkflowEngine `_handle_fast_reply()`
- ✅ **Updated all test files** to remove execution type dependencies
- ✅ **Created new test suite** `tests/unit/test_lifecycle_system_cleanup.py` (8/8 tests passing)
- ✅ **Replaced complex test** with simplified `tests/unit/test_tool_handling.py`

### **Phase 2: Lifecycle System Foundation (PARTIAL)**

- ✅ **Implemented `_load_single_file_lifecycle_functions()`** in RoleRegistry
- ✅ **Lifecycle function discovery working** - verified with both weather and conversation roles
- ⚠️ **Missing**: Complete UniversalAgent lifecycle execution implementation

## **THE CORE PROBLEM IDENTIFIED**

The original issue was **correctly diagnosed**:

1. **Weather role used a hack**: `process_weather_request_with_data._universal_agent = self.universal_agent`
2. **Lifecycle system was broken**: Weather bypassed proper lifecycle execution entirely
3. **No generalized system**: Other roles couldn't use lifecycle hooks

## **ARCHITECTURAL INSIGHTS FROM DOCS 25, 26, 30, 33**

### **Key Principles (LLM-Safe Architecture):**

- **Single Event Loop**: Eliminate threading complexity entirely
- **Intent-Based Processing**: Pure functions returning intents, not complex async chains
- **LLM-Safe Patterns**: Avoid complex async/await propagation
- **Unified Intent Processing**: System already has proper intent processing

### **The Async/Await Problem:**

The implementation got complex because I introduced async/await propagation throughout the system, which **violates the LLM-Safe architecture principles**. The documents clearly state we should eliminate threading complexity, not create more.

## **WHAT NEEDS TO BE DONE NEXT**

### **Phase 2: Complete Lifecycle System (SIMPLE APPROACH)**

1. **Implement `_execute_task_with_lifecycle()` in UniversalAgent**

   - **Keep it SYNC** - no async/await complexity
   - Use `asyncio.run()` only when calling async lifecycle functions
   - Follow the existing pattern from timer role

2. **Fix Weather Role Properly**

   - Add lifecycle config to `ROLE_CONFIG`
   - Standardize function signatures to match lifecycle pattern
   - Remove `process_weather_request_with_data` hack entirely

3. **Fix Conversation Role**
   - Add simple lifecycle functions using Redis tools (like existing helpers)
   - Keep functions simple and sync where possible

### **Phase 3: Test and Verify**

- Test weather role works through standard lifecycle path
- Test conversation role memory continuity
- Verify no special cases remain

## **CRITICAL SUCCESS CRITERIA**

1. **No special cases** in WorkflowEngine for any role
2. **All roles use unified execution** through `_execute_task_with_lifecycle()`
3. **Weather role works** without the current hack
4. **Conversation role works** with proper memory loading/saving
5. **System follows LLM-Safe patterns** - no complex async chains

## **FILES TO FOCUS ON**

### **Primary Files:**

- `llm_provider/universal_agent.py` - Implement `_execute_task_with_lifecycle()`
- `roles/core_weather.py` - Add lifecycle config and fix functions
- `roles/core_conversation.py` - Add simple lifecycle functions

### **Test Files:**

- `tests/unit/test_lifecycle_system_cleanup.py` - Verify architecture
- `tests/unit/test_weather_lifecycle.py` - Test weather lifecycle
- Test conversation memory continuity manually

## **ARCHITECTURAL GUIDANCE**

### **DO:**

- Keep lifecycle execution **SYNC** in the main path
- Use `asyncio.run()` only when calling individual async lifecycle functions
- Follow existing Redis tools patterns (simple and working)
- Use the timer role as a template for lifecycle patterns

### **DON'T:**

- Create complex async/await chains throughout the system
- Use complex memory provider APIs when Redis tools work fine
- Add execution type complexity back
- Create special cases for specific roles

## **VERIFICATION COMMANDS**

```bash
# Test lifecycle function discovery
python -c "
from llm_provider.role_registry import RoleRegistry
registry = RoleRegistry('roles')
print('Weather functions:', list(registry.get_lifecycle_functions('weather').keys()))
print('Conversation functions:', list(registry.get_lifecycle_functions('conversation').keys()))
"

# Test memory continuity
python cli.py --workflow "I'm 6'4\""
python cli.py --workflow "Do you remember my height?"
```

## **NEXT STEPS FOR NEW LLM SESSION**

1. **Implement missing `_execute_task_with_lifecycle()` method** in UniversalAgent
2. **Add lifecycle config to weather role** and fix function signatures
3. **Add simple lifecycle functions to conversation role** using Redis tools
4. **Test end-to-end functionality** and verify memory continuity
5. **Clean up any remaining technical debt**

The foundation is solid - just need to complete the lifecycle execution implementation following LLM-Safe patterns.
