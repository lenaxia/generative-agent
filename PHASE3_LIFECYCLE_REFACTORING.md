# Phase 3 Lifecycle Refactoring

## Summary

Refactored Phase 3 domain roles from independent async execution to lifecycle-compatible pattern that integrates with UniversalAgent's lifecycle for efficient agent pooling.

## Problem

Original Phase 3 domain roles had these issues:

1. **Independent Execution**: Roles had their own `execute()` method that created separate Agent instances
2. **Bypassed Agent Pooling**: Each execution created a new Agent, wasting resources
3. **Async/Sync Conflict**: Domain roles were async, incompatible with UniversalAgent's sync lifecycle
4. **No Lifecycle Hooks**: Couldn't use pre-processors, post-processors, or save functions
5. **Production Gap**: UniversalAgent.assume_role() only checked old roles, never used domain roles

## Solution: Lifecycle-Compatible Pattern

Domain roles now provide **configuration** for UniversalAgent rather than executing independently.

### What Changed

#### Before (Old Pattern)
```python
class WeatherRole:
    REQUIRED_TOOLS = ["weather.get_current_weather", "weather.get_forecast"]

    async def initialize(self):
        self.tools = self.tool_registry.get_tools(self.REQUIRED_TOOLS)

    async def execute(self, request: str) -> str:
        # Create IntentCollector
        # Create Agent
        # Execute with Agent
        # Return result
```

#### After (Lifecycle-Compatible Pattern)
```python
class WeatherRole:
    REQUIRED_TOOLS = ["weather.get_current_weather", "weather.get_forecast"]

    async def initialize(self):
        self.tools = self.tool_registry.get_tools(self.REQUIRED_TOOLS)

    def get_system_prompt(self) -> str:
        """Return specialized system prompt for weather queries."""
        return """You are a weather specialist..."""

    def get_llm_type(self) -> LLMType:
        """Return preferred LLM type for this role."""
        return LLMType.WEAK

    def get_tools(self):
        """Return loaded tools."""
        return self.tools
```

### Key Changes

**Removed:**
- ❌ `async def execute()` method
- ❌ IntentCollector creation/management
- ❌ Agent instance creation
- ❌ Direct LLM interaction

**Added:**
- ✅ `get_system_prompt()` - Returns role-specific system prompt
- ✅ `get_llm_type()` - Returns preferred LLM type (WEAK/DEFAULT/STRONG)

**Kept:**
- ✅ `REQUIRED_TOOLS` - Declares which tools the role needs
- ✅ `async def initialize()` - Loads tools from registry
- ✅ `get_tools()` - Returns loaded tool instances

### UniversalAgent Integration

UniversalAgent.assume_role() now checks for domain roles first:

```python
# Check for Phase 3 domain role
domain_role = self.role_registry.get_domain_role(role)
if domain_role:
    logger.info(f"✨ Using Phase 3 domain role: {role}")

    # Extract configuration
    llm_type = domain_role.get_llm_type()
    tools = domain_role.get_tools()
    system_prompt = domain_role.get_system_prompt()

    # Create role_def wrapper for compatibility
    role_def = create_role_def_wrapper(domain_role)

    # Execute through normal lifecycle with agent pooling
    return self._execute_task_with_lifecycle(...)
```

## Benefits

1. **Agent Pooling**: Domain roles benefit from efficient agent reuse
2. **Lifecycle Hooks**: Can use pre-processors, post-processors, save functions
3. **Consistency**: All roles execute through same path
4. **Simplicity**: Roles only provide configuration, not execution logic
5. **Performance**: No duplicate Agent creation

## Refactored Roles

All 4 Phase 3 domain roles have been refactored:

- ✅ `roles/weather/role.py` - LLMType.WEAK (simple queries)
- ✅ `roles/calendar/role.py` - LLMType.DEFAULT (calendar operations)
- ✅ `roles/timer/role.py` - LLMType.WEAK (simple operations)
- ✅ `roles/smart_home/role.py` - LLMType.DEFAULT (home control)

## Verification

Run the lifecycle integration test:
```bash
python3 test_phase3_lifecycle_integration.py
```

Test in production:
```bash
python3 cli.py

# Try these queries:
> whats the weather in seattle?
> set a timer for 5 minutes
> whats on my calendar today?
> turn on the living room lights
```

Look for log messages like:
```
INFO - ✨ Using Phase 3 domain role: weather with 2 tools
INFO - ⚡ Switched to role 'weather' with 2 tools
```

## Migration Guide

To convert a domain role to lifecycle-compatible pattern:

1. **Remove execute() method**
   ```python
   # Delete this entire method
   async def execute(self, request: str) -> str:
       ...
   ```

2. **Add get_system_prompt() method**
   ```python
   def get_system_prompt(self) -> str:
       """Provide role-specific system prompt."""
       return """You are a [role name] specialist.

       Use the available tools to [describe purpose].

       Available tools:
       - [list tools with brief descriptions]

       [Any special instructions]
       """
   ```

3. **Add get_llm_type() method**
   ```python
   def get_llm_type(self) -> LLMType:
       """Choose based on complexity:
       - WEAK: Simple, deterministic operations
       - DEFAULT: Moderate complexity
       - STRONG: Complex reasoning required
       """
       return LLMType.WEAK  # or DEFAULT/STRONG
   ```

4. **Keep existing methods**
   - REQUIRED_TOOLS declaration
   - async def initialize()
   - def get_tools()

## Implementation Details

### Role Definition Wrapper

UniversalAgent creates a compatibility wrapper to bridge domain roles with existing lifecycle code:

```python
role_def = type('obj', (object,), {
    'config': {
        'type': 'domain_based',
        'name': role,
        'prompts': {'system': domain_role.get_system_prompt()},
    },
    'custom_tools': domain_role.get_tools(),
    'shared_tools': {}
})()
```

This allows domain roles to work with code that expects RoleDefinition objects.

### Old Role Conflicts

The test detected conflicts with old single-file roles:
```
⚠ weather: conflicts with old role pattern
⚠ calendar: conflicts with old role pattern
⚠ timer: conflicts with old role pattern
⚠ smart_home: conflicts with old role pattern
```

This is not a blocker because UniversalAgent checks domain roles first. However, we should eventually remove the old single-file versions to avoid confusion.

## Next Steps

1. ✅ Refactor all domain roles to lifecycle-compatible pattern
2. ✅ Update UniversalAgent to use domain role configuration
3. ✅ Create lifecycle integration test
4. ⏳ Test all roles individually in production
5. ⏹ Remove old single-file role versions if they conflict
6. ⏹ Update validation scripts
7. ⏹ Document pattern for future domain roles

## Production Testing

Use these test queries to verify each role:

**Weather:**
```
> whats the weather in seattle?
> give me a 7 day forecast for new york
```

**Calendar:**
```
> whats on my calendar today?
> add a meeting tomorrow at 2pm
```

**Timer:**
```
> set a timer for 5 minutes
> set a 30 second timer for tea
```

**Smart Home:**
```
> turn on the living room lights
> set the bedroom thermostat to 72 degrees
```

Each should show:
- `✨ Using Phase 3 domain role: [role_name] with [N] tools`
- Correct tools from ToolRegistry (not calculator/file/shell)
- Proper LLM selection (WEAK or DEFAULT)
