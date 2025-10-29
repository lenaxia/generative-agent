# Timer Expiry Notification Fix

## Issue

Timer expiry notifications were not being sent to users. Timers were created successfully and stored in Redis, but users never received notifications when timers expired.

## Root Causes

There were TWO issues preventing timer expiry notifications:

### Issue 1: Heartbeat Tasks Not Starting

The `_start_scheduled_tasks()` method in `supervisor/supervisor.py` was not actually starting the heartbeat tasks. The method only logged a message but never called `asyncio.create_task()` to schedule the async heartbeat coroutines.

### Issue 2: Intent Handlers Not Registered

The `set_intent_processor()` method in `llm_provider/role_registry.py` was not re-registering role intent handlers when the IntentProcessor was set. This meant timer intent handlers (including `TimerCreationIntent` and `TimerExpiryIntent`) were never registered with the IntentProcessor.

### The Problem Chain

1. **Timer Creation** ⚠️ - Tool called but intent handler not registered
2. **Intent Processing** ❌ - `TimerCreationIntent` handler missing, timer not actually created in Redis
3. **Heartbeat Tasks** ❌ - Never started because `_start_scheduled_tasks()` was empty
4. **Fast Heartbeat** ❌ - `_create_fast_heartbeat_task()` never scheduled, so no `FAST_HEARTBEAT_TICK` events published
5. **Timer Monitoring** ❌ - `handle_heartbeat_monitoring()` never triggered (subscribes to `FAST_HEARTBEAT_TICK`)
6. **Expiry Detection** ❌ - Expired timers in Redis never detected (even if they existed)
7. **Expiry Intent** ❌ - `TimerExpiryIntent` handler also not registered
8. **Notifications** ❌ - Users never notified

## The Fixes

### Fix 1: Start Heartbeat Tasks

Modified `supervisor/supervisor.py` line 754-764 to actually start the heartbeat tasks:

```python
def _start_scheduled_tasks(self):
    """Start scheduled tasks for heartbeat operations."""
    try:
        loop = asyncio.get_running_loop()
        # Create and schedule the heartbeat tasks
        asyncio.create_task(self._create_heartbeat_task())
        asyncio.create_task(self._create_fast_heartbeat_task())
        logger.info("Heartbeat tasks started successfully in event loop")
    except RuntimeError:
        # No event loop available yet - will be started when async context is available
        logger.warning("No event loop available - heartbeat tasks will start with async context")
```

**Before:** Method only logged, never started tasks
**After:** Method calls `asyncio.create_task()` for both heartbeat coroutines

### Fix 2: Register Intent Handlers

Modified `llm_provider/role_registry.py` line 98-108 to re-register all role intents when IntentProcessor is set:

```python
def set_intent_processor(self, intent_processor):
    """Set the intent processor for role intent handler registration."""
    self.intent_processor = intent_processor
    logger.info("IntentProcessor set on RoleRegistry")

    # Re-register all single-file role intents now that we have an IntentProcessor
    self._register_all_single_file_role_intents()
```

**Before:** Method only set the processor, never registered handlers
**After:** Method calls `_register_all_single_file_role_intents()` to register all role intent handlers

## How It Works Now

1. When `supervisor.start()` is called, it calls `_start_scheduled_tasks()`
2. `_start_scheduled_tasks()` now schedules both heartbeat tasks in the event loop
3. `_create_fast_heartbeat_task()` publishes `FAST_HEARTBEAT_TICK` every 5 seconds
4. `handle_heartbeat_monitoring()` in `core_timer.py` receives these events
5. It checks Redis for expired timers and creates `TimerExpiryIntent` objects
6. `process_timer_expiry_intent()` sends notifications via `IntentProcessor`
7. Users receive timer expiry notifications through their channels

## Testing

- ✅ `tests/unit/test_supervisor_scheduled_tasks.py` - All 10 tests pass
- ✅ `tests/integration/test_threading_fixes.py::test_timer_handler_returns_intents` - Passes
- ✅ `tests/integration/test_threading_fixes.py::test_timer_role_single_file_structure` - Passes
- ✅ `tests/integration/test_timer_notification_routing.py` - 7 of 8 tests pass (1 needs pytest-asyncio)

## Files Modified

- `supervisor/supervisor.py` - Fixed `_start_scheduled_tasks()` method (lines 754-764)
- `llm_provider/role_registry.py` - Fixed `set_intent_processor()` method (lines 98-108)

## Related Components

- `supervisor/supervisor.py:730-752` - `_create_fast_heartbeat_task()` coroutine
- `supervisor/supervisor.py:710-728` - `_create_heartbeat_task()` coroutine
- `roles/core_timer.py:216-273` - `handle_heartbeat_monitoring()` event handler
- `roles/core_timer.py:557-594` - `process_timer_expiry_intent()` notification sender
- `common/message_bus.py:107` - `FAST_HEARTBEAT_TICK` event registration

## Impact

This fix restores timer expiry notifications for all channels (Slack, console, etc.). The heartbeat monitoring system now runs as designed, checking for expired timers every 5 seconds and notifying users when their timers expire.
