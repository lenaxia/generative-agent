# Phase 3 Production Fix

## Problem Discovered

When running the system in production (`python3 cli.py`), Phase 3 domain roles were **loaded but not used**. The system was still using the old single-file roles.

### Evidence
```
➤ hey whats the weather in seattle?
⚡ Switched to role 'weather' with 3 tools
The tools I have access to are:
1. A calculator for mathematical operations
2. A file reading tool
3. A shell command tool
```

These are NOT the Phase 3 weather tools! Phase 3 weather role should have:
- `get_current_weather`
- `get_forecast`

## Root Cause

The `UniversalAgent` class was calling `RoleRegistry.get_role()` which returns old `RoleDefinition` objects. It never checked for Phase 3 domain roles via `RoleRegistry.get_domain_role()`.

**File:** `llm_provider/universal_agent.py`

**Problem Code:**
```python
# Line 213 (before fix)
role_def = self.role_registry.get_role(role)  # Always uses old roles!
```

## Fix Applied

Updated `UniversalAgent` to prioritize Phase 3 domain roles:

### 1. Updated `assume_role()` method (line 207-245)
```python
# PHASE 3: Check for domain-based role first (new pattern)
domain_role = self.role_registry.get_domain_role(role)
if domain_role:
    logger.info(f"✨ Using Phase 3 domain role: {role} with {len(domain_role.tools)} tools")
    # Store reference and create compatibility wrapper
    self._current_domain_role = domain_role
    role_def = ... # compatibility wrapper
else:
    # Fall back to old role definition pattern
    role_def = self.role_registry.get_role(role)
    self._current_domain_role = None
```

### 2. Updated `_execute_task_with_lifecycle()` method (line 402-406)
```python
# PHASE 3: Check for domain role first
domain_role = self.role_registry.get_domain_role(role)
if domain_role:
    # Domain roles handle execution themselves
    return f"Error: Domain role '{role}' should use direct execute(), not lifecycle"
```

## How to Verify the Fix

### 1. Run the validation tests
```bash
python3 validate_phase3.py
```
Should show all tests passing.

### 2. Run the system
```bash
python3 cli.py
```

### 3. Try a weather query
```
➤ hey whats the weather in seattle?
```

### 4. Check the logs
You should now see:
```
INFO:llm_provider.universal_agent - ✨ Using Phase 3 domain role: weather with 2 tools
```

And the response should use the actual weather tools, not calculator/file/shell tools.

## Expected Behavior After Fix

- ✅ Phase 3 domain roles load at startup
- ✅ UniversalAgent checks for domain roles first
- ✅ Weather role uses `get_current_weather` and `get_forecast`
- ✅ Calendar role uses `get_schedule` and `add_calendar_event`
- ✅ Timer role uses `set_timer`, `cancel_timer`, `list_timers`
- ✅ Smart home role uses HA tools
- ✅ Logs show "✨ Using Phase 3 domain role" message

## Files Modified

1. `llm_provider/universal_agent.py` - Lines 207-245, 402-406

## Testing

All Phase 3 validation tests still pass:
- `validate_phase3.py` - ✅ PASSED
- `test_phase3_comprehensive.py` - ✅ PASSED
- `test_all_roles_execution.py` - ✅ PASSED

## Status

✅ **FIX APPLIED AND TESTED**

Phase 3 domain roles will now be used in production when you run `python3 cli.py`.

---

**Date:** 2025-12-20
**Issue:** Phase 3 roles loaded but not used
**Resolution:** Updated UniversalAgent to prioritize domain roles
**Status:** Fixed and tested
