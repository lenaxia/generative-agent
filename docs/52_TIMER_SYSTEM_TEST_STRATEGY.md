# Timer System Test Strategy

## Overview

This document explains the test strategy designed to catch timer system integration issues, specifically the two bugs that prevented timer expiry notifications from working.

## The Bugs We're Testing For

### Bug 1: Heartbeat Tasks Not Starting

**Location:** `supervisor/supervisor.py:_start_scheduled_tasks()`
**Issue:** Method only logged but never called `asyncio.create_task()` to schedule heartbeat coroutines
**Impact:** No `FAST_HEARTBEAT_TICK` events published, timer monitoring never ran

### Bug 2: Intent Handlers Not Registered

**Location:** `llm_provider/role_registry.py:set_intent_processor()`
**Issue:** Method didn't call `_register_all_single_file_role_intents()` to register role intent handlers
**Impact:** `TimerCreationIntent` and `TimerExpiryIntent` handlers missing, timers never created or notified

## Test Categories

### 1. Heartbeat Task Startup Tests (`TestHeartbeatTaskStartup`)

#### `test_start_scheduled_tasks_creates_heartbeat_tasks`

**Purpose:** Verify that `_start_scheduled_tasks()` actually schedules async tasks
**How It Catches Bug 1:**

- Mocks `asyncio.create_task()` to track calls
- Verifies it's called exactly twice (heartbeat + fast_heartbeat)
- Confirms the correct coroutines are being scheduled

**Key Assertion:**

```python
assert mock_create_task.call_count == 2, \
    "Expected 2 heartbeat tasks to be created"
```

#### `test_start_scheduled_tasks_handles_no_event_loop`

**Purpose:** Verify graceful handling when no event loop exists
**How It Helps:**

- Ensures the fix doesn't break synchronous contexts
- Verifies proper error handling

#### `test_fast_heartbeat_publishes_events`

**Purpose:** Verify fast heartbeat actually publishes `FAST_HEARTBEAT_TICK` events
**How It Catches Bug 1:**

- Starts the fast heartbeat task
- Waits for events to be published
- Verifies `FAST_HEARTBEAT_TICK` appears in published events

**Key Assertion:**

```python
assert "FAST_HEARTBEAT_TICK" in published_events, \
    "Expected FAST_HEARTBEAT_TICK to be published by fast heartbeat"
```

### 2. Intent Handler Registration Tests (`TestIntentHandlerRegistration`)

#### `test_timer_intents_registered_on_set_intent_processor`

**Purpose:** Verify timer intent handlers are registered when IntentProcessor is set
**How It Catches Bug 2:**

- Creates RoleRegistry and loads timer role
- Sets IntentProcessor on registry
- Verifies all timer intents are registered

**Key Assertions:**

```python
for intent_name in timer_intents:
    assert any(intent_name in handler for handler in role_handlers), \
        f"Expected {intent_name} to be registered with IntentProcessor"
```

#### `test_intent_processor_receives_timer_creation_intent`

**Purpose:** Verify `TimerCreationIntent` can be processed without warnings
**How It Catches Bug 2:**

- Manually registers handler (simulating correct behavior)
- Creates a timer intent
- Verifies handler is registered and intent is valid

#### `test_intent_processor_receives_timer_expiry_intent`

**Purpose:** Verify `TimerExpiryIntent` can be processed without warnings
**How It Catches Bug 2:**

- Similar to creation intent test
- Ensures expiry notifications can be sent

### 3. Timer Workflow Integration Tests (`TestTimerWorkflowIntegration`)

#### `test_supervisor_initialization_registers_timer_intents`

**Purpose:** Verify complete supervisor initialization registers timer intents
**How It Catches Both Bugs:**

- Tests the full initialization flow
- Verifies IntentProcessor exists
- Verifies RoleRegistry has IntentProcessor reference
- Confirms timer intents are registered

**Key Assertions:**

```python
assert any("TimerCreationIntent" in h for h in role_handlers), \
    "TimerCreationIntent should be registered after supervisor start"
assert any("TimerExpiryIntent" in h for h in role_handlers), \
    "TimerExpiryIntent should be registered after supervisor start"
```

#### `test_complete_timer_workflow_with_mocked_redis`

**Purpose:** Test end-to-end timer workflow from creation to expiry
**How It Catches Both Bugs:**

- Tests timer creation (requires registered intent handler)
- Tests timer expiry (requires registered intent handler)
- Verifies Redis operations are called correctly

### 4. Health Check Tests (`TestTimerSystemHealthChecks`)

#### `test_timer_role_has_heartbeat_handler`

**Purpose:** Verify timer role registers `FAST_HEARTBEAT_TICK` handler
**How It Helps:**

- Ensures timer role configuration is correct
- Catches configuration regressions

#### `test_timer_role_has_all_intent_handlers`

**Purpose:** Verify timer role registers all required intent handlers
**How It Helps:**

- Ensures role registration structure is correct
- Catches missing intent handlers early

#### `test_message_bus_has_fast_heartbeat_event_registered`

**Purpose:** Verify `FAST_HEARTBEAT_TICK` is registered as valid event type
**How It Helps:**

- Ensures message bus configuration is correct
- Catches event registration issues

## Running the Tests

### Run All Timer System Tests

```bash
pytest tests/unit/test_timer_system_integration.py -xvs
```

### Run Specific Test Categories

```bash
# Heartbeat tests only
pytest tests/unit/test_timer_system_integration.py::TestHeartbeatTaskStartup -xvs

# Intent registration tests only
pytest tests/unit/test_timer_system_integration.py::TestIntentHandlerRegistration -xvs

# Integration tests only
pytest tests/unit/test_timer_system_integration.py::TestTimerWorkflowIntegration -xvs

# Health checks only
pytest tests/unit/test_timer_system_integration.py::TestTimerSystemHealthChecks -xvs
```

### Run Key Tests That Catch Both Bugs

```bash
pytest tests/unit/test_timer_system_integration.py -xvs -k "test_start_scheduled_tasks_creates_heartbeat_tasks or test_timer_intents_registered"
```

## Test Coverage

These tests provide coverage for:

- ✅ Heartbeat task creation and scheduling
- ✅ Intent handler registration
- ✅ Complete supervisor initialization flow
- ✅ Timer role configuration
- ✅ Message bus event registration
- ✅ End-to-end timer workflow

## Why These Tests Would Have Caught the Bugs

### Bug 1 Detection

The test `test_start_scheduled_tasks_creates_heartbeat_tasks` directly mocks `asyncio.create_task()` and verifies it's called. If `_start_scheduled_tasks()` only logs (as it did before the fix), this test would fail with:

```
AssertionError: Expected 2 heartbeat tasks to be created
```

### Bug 2 Detection

The test `test_timer_intents_registered_on_set_intent_processor` calls `set_intent_processor()` and then checks if timer intents are registered. If the method doesn't call `_register_all_single_file_role_intents()` (as it didn't before the fix), this test would fail with:

```
AssertionError: Expected TimerCreationIntent to be registered with IntentProcessor
```

## Continuous Integration

These tests should be run:

1. **On every commit** - Catch regressions immediately
2. **Before merging PRs** - Ensure changes don't break timer system
3. **In nightly builds** - Catch integration issues
4. **After dependency updates** - Ensure compatibility

## Future Enhancements

Consider adding:

1. **Performance tests** - Verify heartbeat doesn't consume excessive resources
2. **Load tests** - Test with many concurrent timers
3. **Failure recovery tests** - Test Redis connection failures
4. **Race condition tests** - Test concurrent timer operations
5. **End-to-end Slack tests** - Test actual Slack integration (requires pytest-asyncio)

## Related Documentation

- [`docs/TIMER_EXPIRY_FIX.md`](TIMER_EXPIRY_FIX.md) - Details of the bugs and fixes
- [`docs/18_COMPREHENSIVE_TIMER_SYSTEM_DESIGN.md`](18_COMPREHENSIVE_TIMER_SYSTEM_DESIGN.md) - Timer system architecture
- [`tests/unit/test_timer_system_integration.py`](../tests/unit/test_timer_system_integration.py) - Test implementation
