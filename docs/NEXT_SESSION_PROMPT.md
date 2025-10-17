# Lifecycle Hook System Implementation - Next Session Prompt

## **TASK FOR NEXT LLM SESSION:**

The lifecycle hook system in this Universal Agent repository was broken - the weather role used hacks instead of proper lifecycle execution, and the conversation role couldn't use lifecycle hooks at all.

**CURRENT STATUS:** Phase 1 is complete (commit 95a7579) - I've removed all execution type complexity and created a clean foundation. The system is stable but the lifecycle implementation needs to be completed.

## **WHAT YOU NEED TO DO:**

1. **Read the handoff document**: `LIFECYCLE_HOOK_IMPLEMENTATION_HANDOFF.md` for complete context

2. **Complete Phase 2**: Implement the missing lifecycle execution system following LLM-Safe architecture principles

3. **Key files to focus on**:
   - `llm_provider/universal_agent.py` - Add missing `_execute_task_with_lifecycle()` method
   - `roles/core_weather.py` - Add lifecycle config and fix function signatures
   - `roles/core_conversation.py` - Add simple lifecycle functions using Redis tools

## **CRITICAL ARCHITECTURAL GUIDANCE:**

### **Follow LLM-Safe Patterns (Documents 25, 26, 30, 33):**

- **Single Event Loop**: No complex async/await chains
- **Keep it SYNC**: Use `asyncio.run()` only when calling individual async lifecycle functions
- **Intent-Based Processing**: Pure functions returning intents
- **No Special Cases**: All roles use unified execution path

### **DO NOT:**

- Create complex async/await propagation throughout the system
- Use complex memory provider APIs when Redis tools work fine
- Add execution type complexity back
- Create special cases for specific roles

## **SUCCESS CRITERIA:**

1. **Weather role works** without the `process_weather_request_with_data` hack
2. **Conversation role works** with proper memory loading/saving
3. **Memory continuity functional** - conversation role remembers previous interactions
4. **No special cases** in WorkflowEngine for any role
5. **All roles use unified execution** through `_execute_task_with_lifecycle()`

## **VERIFICATION COMMANDS:**

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

## **FOUNDATION IS SOLID:**

The hard work is done - execution type complexity is removed, architecture is clean, and RoleRegistry can discover lifecycle functions. You just need to complete the implementation following the simple patterns established.

**Start by reading the handoff document, then implement the missing `_execute_task_with_lifecycle()` method in UniversalAgent.**
