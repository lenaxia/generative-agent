# Timer Expiry Issue - Root Cause Analysis

## Problem
Timers are created successfully and stored in Redis, but they never expire and trigger notifications.

## Evidence
```bash
# Timer created at 20:22:11
Timer created: timer_b2185bf0

# 30 seconds later, timer still in Redis (should have expired)
redis-cli ZRANGE timer:active_queue 0 -1 WITHSCORES
timer_b2185bf0
1766290961.028887  # Expiry timestamp
```

## Architecture
Timers rely on a **heartbeat-driven expiry system**:

1. **Timer Creation**: Timer stored in Redis sorted set with expiry timestamp
2. **Heartbeat Loop**: Background task checks for expired timers every 5 seconds
3. **Expiry Detection**: Queries Redis for timers with timestamp < current_time
4. **Intent Creation**: Creates `TimerExpiryIntent` for each expired timer
5. **Intent Processing**: Sends notification and executes deferred workflow

## Root Cause: Heartbeat May Not Be Running

### The Heartbeat System

**File**: `supervisor/supervisor.py`

**Key Components**:
```python
# Creates heartbeat task (line 830)
async def _create_fast_heartbeat_task(self):
    while True:
        # Only publishes if message bus is running
        if self.message_bus and self.message_bus.is_running():
            self.message_bus.publish(
                message_type="FAST_HEARTBEAT_TICK",
                message={"tick": tick_count, "timestamp": time.time()},
            )
        await asyncio.sleep(5)  # 5 second interval

# Starts heartbeat task (line 854)
def _start_scheduled_tasks(self):
    loop = asyncio.get_running_loop()
    asyncio.create_task(self._create_fast_heartbeat_task())
    logger.info("Heartbeat tasks started successfully in event loop")

# Called from async context (line 588)
async def start_async_tasks(self):
    self._start_scheduled_tasks()
    logger.info("Async heartbeat tasks started successfully")
```

**File**: `roles/core_timer.py`

**Event Handler**:
```python
# Subscribed to FAST_HEARTBEAT_TICK (line 678)
"event_handlers": {
    "FAST_HEARTBEAT_TICK": handle_heartbeat_monitoring,
}

# Handler function (line 226)
def handle_heartbeat_monitoring(event_data, context):
    current_time = int(time.time())
    expired_timer_ids = _get_expired_timers_from_redis(current_time)
    # Create TimerExpiryIntent for each expired timer
    return [TimerExpiryIntent(...) for timer_id in expired_timer_ids]
```

### Why It Might Not Be Working

**Hypothesis 1: Message Bus Not Running**
- Heartbeat only publishes if `message_bus.is_running()` returns True
- Check: Message bus starts in `Supervisor.start()` (line 566)
- **Likely OK** - cli.py calls `supervisor.start()` (line 642)

**Hypothesis 2: Heartbeat Task Not Created**
- Task created in `_start_scheduled_tasks()` (line 860)
- Called from `start_async_tasks()` (line 596)
- **Likely OK** - cli.py calls `await supervisor.start_async_tasks()` (line 645)

**Hypothesis 3: Event Not Being Delivered**
- Message bus publishes event (line 837)
- Timer role subscribed via `register_role()` (line 678)
- **Possible Issue** - Role subscription happens at role registry initialization
- May not be connected to message bus properly

**Hypothesis 4: Logging Level Hides Evidence**
- Heartbeat logs at DEBUG level (line 843-845)
- Default logging might not show these messages
- **Diagnostic Needed** - Run with DEBUG logging

## Diagnostic Steps

### 1. Check if Heartbeat is Publishing

Add this at the start of your CLI session:

```python
# In cli.py, before interactive loop
supervisor.message_bus.subscribe(
    "FAST_HEARTBEAT_TICK",
    lambda msg: print(f"❤️ Heartbeat tick {msg.get('tick')}")
)
```

### 2. Check if Timer Role is Receiving Events

```bash
# Run with debug logging
python3 cli.py --log-level DEBUG 2>&1 | grep -E "(FAST_HEARTBEAT|heartbeat|expired timer)"
```

### 3. Manually Trigger Heartbeat Handler

```python
# Test the handler function directly
from roles.core_timer import handle_heartbeat_monitoring
intents = handle_heartbeat_monitoring({}, None)
print(f"Found {len(intents)} expired timers")
```

### 4. Check Message Bus Subscribers

```python
# After supervisor starts
print(f"Subscribers to FAST_HEARTBEAT_TICK: {
    len(supervisor.message_bus._subscribers.get('FAST_HEARTBEAT_TICK', []))
}")
```

## Quick Fix: Manual Heartbeat Test

Create `test_timer_expiry.py`:

```python
import asyncio
import time
from supervisor.supervisor import Supervisor

async def test():
    supervisor = Supervisor(None)
    supervisor.start()
    await supervisor.start_async_tasks()

    # Subscribe to heartbeat
    tick_count = [0]
    def on_heartbeat(msg):
        tick_count[0] += 1
        print(f"Tick {tick_count[0]}: {msg}")

    supervisor.message_bus.subscribe("FAST_HEARTBEAT_TICK", on_heartbeat)

    # Wait for 3 ticks
    print("Waiting 16 seconds for 3 heartbeat ticks...")
    await asyncio.sleep(16)

    if tick_count[0] >= 3:
        print(f"✓ Heartbeat working ({tick_count[0]} ticks)")
    else:
        print(f"✗ Heartbeat NOT working ({tick_count[0]} ticks)")

asyncio.run(test())
```

## Possible Solutions

### Solution 1: Verify Subscription
The timer role might not be properly subscribing to heartbeat events.

**Check**: In `RoleRegistry.register_role_def()`, ensure event handlers are registered with message bus.

### Solution 2: Ensure Event Loop Runs
The heartbeat task needs the event loop to actually execute.

**Check**: In `run_interactive_mode()`, ensure `await asyncio.to_thread(input, ...)` allows other tasks to run.

### Solution 3: Add Explicit Heartbeat Start Log
Add more verbose logging to confirm heartbeat starts:

```python
# In _create_fast_heartbeat_task(), add at start:
logger.warning("FAST HEARTBEAT TASK STARTED - will tick every 5 seconds")
```

### Solution 4: Force Synchronous Check (Workaround)
If heartbeat system is broken, add manual check after timer creation:

```python
# In timer creation, schedule a one-time check
asyncio.create_task(check_timer_expiry_later(timer_id, duration_seconds))
```

## Next Steps

1. **Enable Debug Logging**: Run `python3 cli.py` with verbose logging
2. **Check Heartbeat Ticks**: Look for "Fast heartbeat tick" messages
3. **Check Subscriptions**: Verify timer role is subscribed
4. **Test Manually**: Run heartbeat handler function directly

## Expected Log Messages

When working correctly, you should see:

```
INFO - Heartbeat tasks started successfully in event loop
DEBUG - Fast heartbeat tick 1 published (5s interval)
DEBUG - Checking for expired timers at time: 1766290961
INFO - Found 1 expired timers: ['timer_b2185bf0']
DEBUG - Processing expired timer: timer_b2185bf0
INFO - Timer timer_b2185bf0 expired
```

If you don't see these, the heartbeat isn't running.
