# Heartbeat System Diagnostic Results - UPDATED

## Summary

**Status**: Timer storage is **WORKING** when using the virtual environment. Next step is to verify heartbeat expiry detection.

---

## ✅ CONFIRMED Working

### 1. Virtual Environment with Redis Library
```bash
source venv/bin/activate
pip list | grep redis
# redis 6.4.0 ✅
# aioredis 2.0.0 ✅
```

### 2. Timer Creation and Storage
User confirmed timer successfully stored in Redis:
```
timer_54db89e2
1.7662855149038117e+9
```

Logs show successful creation:
```
2025-12-20 22:31:13,943 - roles.timer.tools - INFO - Setting timer for 10s with label: 10 second timer
2025-12-20 22:31:13,943 - roles.timer.tools - INFO - Timer created: timer_92ecce3e
2025-12-20 22:31:13,944 - roles.core_timer - INFO - Creating timer timer_92ecce3e for 10s
2025-12-20 22:31:13,944 - roles.core_timer - DEBUG - Storing timer data for timer_92ecce3e
```

###3. Heartbeat System Initialization
```
2025-12-20 22:30:51,885 - common.intent_processor - INFO - Registered TimerExpiryIntent handler for timer role
2025-12-20 22:30:51,914 - supervisor - INFO - Async heartbeat tasks started successfully
```

---

## ❓ NEEDS VERIFICATION

### Does Heartbeat Detect and Expire Timers?

**What to look for in logs** (with `--verbose`):

1. **Every 5 seconds** - Heartbeat checking:
   ```
   DEBUG - Checking for expired timers at time: 1766298xxx
   ```

2. **When timer expires** - Detection:
   ```
   INFO - Found 1 expired timers: ['timer_54db89e2']
   INFO - Timer timer_54db89e2 expiring for user <user_id> in channel <channel_id>
   INFO - Processing 1 timer expiry notifications
   ```

3. **After expiry** - Intent processing:
   ```
   INFO - Processing intent: TimerExpiryIntent(timer_id='timer_54db89e2', ...)
   ```

---

## Test Procedure

### Run this command and wait 15+ seconds:
```bash
source venv/bin/activate
python cli.py --workflow "set a timer for 10 seconds" --verbose 2>&1 | tee timer_test.log
```

### While running, in another terminal watch Redis:
```bash
watch -n 1 'redis-cli ZRANGE timer:active_queue 0 -1 WITHSCORES'
```

### Expected behavior:
1. **T+0s**: Timer appears in Redis sorted set
2. **T+5s**: Heartbeat tick (check logs for "Checking for expired timers")
3. **T+10s**: Timer expiry time reached
4. **T+10-15s**: Heartbeat detects expired timer (next tick after expiry)
5. **T+10-15s**: Timer removed from Redis
6. **T+10-15s**: Notification sent to user

### If timer does NOT expire:

**Check 1**: Heartbeat ticks happening?
```bash
grep -i "checking for expired timers" timer_test.log
```
If NO matches: Heartbeat not publishing or timer role not subscribed

**Check 2**: Timer in Redis?
```bash
redis-cli ZRANGE timer:active_queue 0 -1 WITHSCORES
```
If NO timer: Not stored (but we confirmed this works)

**Check 3**: Expiry timestamp correct?
```bash
current_time=$(date +%s)
timer_expiry=$(redis-cli ZSCORE timer:active_queue timer_54db89e2)
echo "Current: $current_time, Expiry: $timer_expiry"
# Expiry should be LESS than current if timer should have expired
```

---

## Architecture Validation

### Code Flow for Timer Expiry:

1. **Heartbeat Task** (`supervisor/supervisor.py:817-845`):
   ```python
   async def _create_fast_heartbeat_task(self):
       while True:
           self.message_bus.publish(
               publisher=self,
               message_type="FAST_HEARTBEAT_TICK",
               message={"tick": tick_count, "timestamp": time.time()},
           )
           await asyncio.sleep(5)  # Every 5 seconds
   ```

2. **Event Handler Registration** (`roles/core_timer.py:678`):
   ```python
   def register_role():
       return {
           "event_handlers": {
               "FAST_HEARTBEAT_TICK": handle_heartbeat_monitoring,  # Subscribed here
           },
       }
   ```

3. **Heartbeat Monitoring** (`roles/core_timer.py:225-289`):
   ```python
   def handle_heartbeat_monitoring(event_data: Any, context) -> list[Intent]:
       current_time = int(time.time())
       logger.debug(f"Checking for expired timers at time: {current_time}")

       expired_timer_ids = _get_expired_timers_from_redis(current_time)
       if expired_timer_ids:
           logger.info(f"Found {len(expired_timer_ids)} expired timers: {expired_timer_ids}")

       # Create TimerExpiryIntent for each
       return [TimerExpiryIntent(...) for timer_id in expired_timer_ids]
   ```

4. **Intent Processing** (`roles/core_timer.py:119-171`):
   ```python
   def process_timer_expiry_intent(intent: TimerExpiryIntent, context):
       # Remove from Redis
       redis_delete(f"timer:active_queue", intent.timer_id)
       redis_delete(f"timer:data:{intent.timer_id}")

       # Notify user (via Slack, etc.)
       # Execute deferred workflow if specified
   ```

---

## Key Logging Points

### To enable DEBUG logs temporarily:

**Option 1**: Modify `roles/core_timer.py` line 233:
```python
# Change from:
logger.debug(f"Checking for expired timers at time: {current_time}")
# To:
logger.info(f"Checking for expired timers at time: {current_time}")
```

**Option 2**: Modify `supervisor/supervisor.py` line 837:
```python
# Change from:
logger.debug(f"Fast heartbeat tick {tick_count} published (5s interval)")
# To:
logger.info(f"Fast heartbeat tick {tick_count} published (5s interval)")
```

**Option 3**: Modify `llm_provider/role_registry.py` line 1116:
```python
# Change from:
logger.debug(f"Registered event handler {event_type} for single-file role {role_name}")
# To:
logger.info(f"Registered event handler {event_type} for single-file role {role_name}")
```

Then rerun the test to see these messages at INFO level.

---

## Diagnostic Questions

### Q1: Is heartbeat task actually running?
**A**: Logs show "Async heartbeat tasks started successfully" ✅

### Q2: Is timer role subscribing to FAST_HEARTBEAT_TICK?
**A**: Needs DEBUG logs to confirm (logs at DEBUG level)

### Q3: Are heartbeat ticks being published?
**A**: Needs DEBUG logs to confirm (logs at DEBUG level)

### Q4: Is handle_heartbeat_monitoring being called?
**A**: Should see "Checking for expired timers" every 5 seconds if working

### Q5: Are expired timers being detected?
**A**: Should see "Found X expired timers" when timer expires

---

## Next Steps

1. **Quick Test** (5 minutes):
   - Run CLI with timer
   - Wait 15 seconds
   - Check logs for "Checking for expired timers" or "Found X expired timers"
   - Check if timer removed from Redis

2. **If Not Working** - Enable DEBUG logs:
   - Temporarily change `logger.debug` to `logger.info` in key locations (listed above)
   - Rerun test
   - Check for:
     - "Fast heartbeat tick" messages
     - "Registered event handler FAST_HEARTBEAT_TICK" message
     - "Checking for expired timers" messages

3. **If Still Not Working** - Deep dive:
   - Verify MessageBus.subscribe() actually called
   - Verify event handler function gets invoked
   - Check if event loop is running properly
   - Verify no exceptions in background task

---

## Expected Outcome

**If everything is working correctly**:
1. Timer stored in Redis ✅ (confirmed)
2. Heartbeat publishes FAST_HEARTBEAT_TICK every 5s
3. Timer role receives events via subscription
4. handle_heartbeat_monitoring() called every 5s
5. Expired timers detected and removed
6. User receives notification

**Current Status**: Steps 1 is confirmed. Steps 2-6 need verification.

---

## Critical Files

- `supervisor/supervisor.py:817-845` - Heartbeat task creation
- `roles/core_timer.py:678` - Event handler registration
- `roles/core_timer.py:225-289` - Heartbeat monitoring logic
- `roles/core_timer.py:119-171` - Timer expiry intent processing
- `llm_provider/role_registry.py:1105-1128` - Event handler subscription

---

## Conclusion

The fix for using the virtual environment was correct. Timer creation and Redis storage are now working. The remaining unknown is whether the heartbeat system successfully detects and expires timers. This can be verified by running the test procedure above and looking for the expected log messages.

