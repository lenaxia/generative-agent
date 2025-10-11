# Timer Notification Threading Fix Design

**Document ID:** 22
**Created:** 2025-10-11
**Status:** Design Phase
**Priority:** Critical

## Executive Summary

This document outlines the design for fixing a critical issue where timer expiry notifications fail to reach communication channels due to threading/event loop conflicts in the aiohttp session management. While initial timer confirmations work perfectly, subsequent timer expiry notifications "black hole" and never complete their HTTP requests.

## Problem Statement

### Current Issue

- ✅ Initial timer confirmation messages work perfectly (237ms response time)
- ✅ Timer expiry events are detected and processed correctly
- ✅ Communication flow reaches SlackHandler with correct payload
- ❌ **Critical Failure**: Timer expiry HTTP requests hang indefinitely
- ❌ No timer expiry notifications reach users

### Impact

- Complete breakdown of timer notification functionality
- Users set timers but never receive expiry notifications
- Silent failure - no error messages, requests just hang
- System appears functional but timer feature is broken

## Root Cause Analysis

### Technical Root Cause

The issue lies in threading/event loop conflicts when aiohttp sessions are created from different thread contexts:

**Working Flow (Initial Response):**

```
Main Supervisor Thread → WorkflowEngine → CommunicationManager → SlackHandler
├─ Proper async context
├─ aiohttp session works correctly
└─ HTTP request completes in 237ms ✅
```

**Failing Flow (Timer Expiry):**

```
Heartbeat Thread → TimerMonitor → MessageBus → CommunicationManager → SlackHandler
├─ Background thread context
├─ Separate event loop thread
├─ aiohttp session created in wrong context
└─ HTTP request hangs indefinitely ❌
```

### Evidence from Logs

```bash
# ✅ Working: Initial timer confirmation
21:32:27,010 - Slack API call starting with payload: {'channel': '#general', 'text': 'Timer set for 10 seconds...'}
21:32:27,247 - Received response with status: 200
21:32:27,248 - Channel slack send_notification result: {'success': True, ...}

# ❌ Failing: Timer expiry notification
21:32:54,056 - Slack API call starting with payload: {'channel': '#general', 'text': '⏰ 10s timer expired!'}
21:32:54,056 - Created aiohttp session, making POST request...
[HANGS HERE - NO RESPONSE LOGGED]
```

### The Problem Chain

1. **TimerMonitor** runs in Heartbeat background thread
2. **CommunicationManager** processes messages in separate event loop thread
3. **SlackHandler.\_send_via_api()** creates new aiohttp session in wrong thread context
4. **aiohttp session** cannot coordinate properly with event loop from different thread
5. **HTTP request hangs** indefinitely (never completes or times out)

### Code Location

The issue occurs in `common/channel_handlers/slack_handler.py:234-244`:

```python
async def _send_via_api(self, ...):
    try:
        timeout = aiohttp.ClientTimeout(total=10.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:  # ← PROBLEM
            async with session.post(...) as response:  # ← HANGS HERE
```

## Solution Design

### Overview

Implement thread-safe HTTP communication that works correctly across different thread contexts while maintaining the existing architecture.

### Solution Options

#### Option 1: Thread-Safe HTTP Client (Recommended - Quick Fix)

**Approach:** Use `requests` library for timer notifications since it's thread-safe.

**Implementation:**

```python
async def _send_via_api_threadsafe(self, message, channel, message_format, buttons, metadata):
    """Thread-safe version using requests library for cross-thread calls."""
    import requests
    import asyncio

    payload = {"channel": channel, "text": message}

    # Add thread_ts, blocks, attachments as before
    if "thread_ts" in metadata:
        payload["thread_ts"] = metadata["thread_ts"]
    if "blocks" in metadata:
        payload["blocks"] = metadata["blocks"]
    elif buttons:
        payload["blocks"] = self._create_blocks_with_buttons(message, buttons)
    if "attachments" in metadata:
        payload["attachments"] = metadata["attachments"]

    try:
        logger.info(f"Thread-safe Slack API call starting with payload: {payload}")

        # Run requests.post in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,  # Use default thread pool
            lambda: requests.post(
                "https://slack.com/api/chat.postMessage",
                json=payload,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {self.bot_token}",
                },
                timeout=10.0
            )
        )

        logger.info(f"Received response with status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            if response_data.get("ok"):
                return {
                    "success": True,
                    "channel": channel,
                    "ts": response_data.get("ts"),
                    "message_id": response_data.get("ts"),
                }
            else:
                return {
                    "success": False,
                    "error": f"Slack API error: {response_data.get('error')}",
                }
        else:
            return {
                "success": False,
                "error": f"Slack API error: {response.status_code} - {response.text}",
            }

    except Exception as e:
        logger.error(f"Error sending thread-safe Slack API request: {str(e)}")
        return {"success": False, "error": str(e)}
```

**Detection Logic:**

```python
def _is_background_thread(self) -> bool:
    """Detect if we're running in a background thread context."""
    import threading
    current_thread = threading.current_thread()
    return current_thread.name != "MainThread"

async def _send_via_api(self, message, channel, message_format, buttons, metadata):
    """Send message with automatic thread-safe detection."""
    if self._is_background_thread():
        logger.info("Background thread detected, using thread-safe HTTP client")
        return await self._send_via_api_threadsafe(message, channel, message_format, buttons, metadata)
    else:
        logger.info("Main thread detected, using aiohttp")
        return await self._send_via_api_aiohttp(message, channel, message_format, buttons, metadata)
```

#### Option 2: Shared Session Pool (Long-term Solution)

**Approach:** Implement shared aiohttp session with proper event loop management.

**Implementation:**

```python
class SlackChannelHandler(ChannelHandler):
    def __init__(self, config):
        super().__init__(config)
        self._session = None
        self._session_lock = None
        self._main_loop = None

    async def _get_or_create_session(self):
        """Get or create shared aiohttp session in main event loop."""
        if self._session is None or self._session.closed:
            if self._session_lock is None:
                self._session_lock = asyncio.Lock()

            async with self._session_lock:
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(total=10.0)
                    self._session = aiohttp.ClientSession(timeout=timeout)

        return self._session

    async def _send_via_shared_session(self, message, channel, message_format, buttons, metadata):
        """Send using shared session with proper event loop coordination."""
        payload = {"channel": channel, "text": message}
        # ... payload setup ...

        try:
            session = await self._get_or_create_session()
            async with session.post(
                "https://slack.com/api/chat.postMessage",
                json=payload,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {self.bot_token}",
                },
            ) as response:
                # ... response handling ...
        except Exception as e:
            # ... error handling ...
```

#### Option 3: Event Loop Bridge (Advanced Solution)

**Approach:** Bridge requests from background threads to main event loop.

**Implementation:**

```python
async def _send_via_loop_bridge(self, message, channel, message_format, buttons, metadata):
    """Bridge HTTP request to main event loop."""
    import asyncio
    import concurrent.futures

    # Get main event loop reference
    main_loop = self._get_main_event_loop()

    if main_loop and main_loop != asyncio.get_event_loop():
        # We're in background thread, bridge to main loop
        future = asyncio.run_coroutine_threadsafe(
            self._send_via_api_aiohttp(message, channel, message_format, buttons, metadata),
            main_loop
        )
        return future.result(timeout=15.0)  # 15 second timeout
    else:
        # We're in main loop, proceed normally
        return await self._send_via_api_aiohttp(message, channel, message_format, buttons, metadata)
```

### Recommended Implementation Plan

#### Phase 1: Quick Fix (Option 1)

1. **Implement thread detection** in SlackHandler
2. **Add thread-safe HTTP client** using requests library
3. **Automatic fallback** based on thread context
4. **Minimal code changes** to existing flow

#### Phase 2: Long-term Solution (Option 2)

1. **Implement shared session pool** for better performance
2. **Proper event loop management** across threads
3. **Connection pooling** and reuse
4. **Enhanced error handling** and recovery

#### Phase 3: Advanced Features (Option 3)

1. **Event loop bridging** for complex scenarios
2. **Performance monitoring** and metrics
3. **Adaptive threading** based on load

## Implementation Details

### Files to Modify

#### 1. `common/channel_handlers/slack_handler.py`

```python
# Add new methods:
- _is_background_thread()
- _send_via_api_threadsafe()
- _send_via_api_aiohttp() (rename existing)

# Modify existing method:
- _send_via_api() (add detection logic)
```

#### 2. `requirements.txt`

```
# Add if not present:
requests>=2.31.0
```

### Configuration Changes

#### `config.yaml`

```yaml
# Add HTTP client configuration
communication:
  http_client:
    thread_safe_fallback: true
    timeout_seconds: 10
    max_retries: 3

  slack:
    use_shared_session: false # Enable in Phase 2
    session_pool_size: 5 # For Phase 2
```

### Testing Strategy

#### Unit Tests

```python
# tests/unit/test_slack_threading_fix.py
class TestSlackThreadingFix:
    def test_background_thread_detection(self):
        """Test thread context detection."""

    def test_thread_safe_http_client(self):
        """Test requests-based HTTP client."""

    def test_main_thread_aiohttp_client(self):
        """Test aiohttp client in main thread."""

    def test_automatic_fallback(self):
        """Test automatic selection based on thread context."""
```

#### Integration Tests

```python
# tests/integration/test_timer_notification_fix.py
class TestTimerNotificationFix:
    def test_timer_expiry_notification_delivery(self):
        """Test end-to-end timer expiry notification."""

    def test_concurrent_notifications(self):
        """Test multiple timer notifications simultaneously."""

    def test_thread_safety_under_load(self):
        """Test thread safety with high notification volume."""
```

### Performance Considerations

#### Metrics to Track

- **HTTP Request Completion Rate**: Should be 100% after fix
- **Response Time**: requests library may be slightly slower than aiohttp
- **Memory Usage**: Monitor for session leaks
- **Thread Safety**: No race conditions or deadlocks

#### Expected Performance Impact

- **Positive**: Timer notifications will actually work (0% → 100% success rate)
- **Minimal**: Slight increase in response time for background thread requests
- **Negligible**: Memory overhead from requests library

### Rollback Plan

#### If Issues Arise

1. **Feature Flag**: Add `enable_threading_fix: false` to disable
2. **Gradual Rollout**: Test with subset of timer notifications first
3. **Monitoring**: Track success rates and error patterns
4. **Quick Revert**: Simple config change to disable new code path

## Risk Assessment

### Low Risk

- **Isolated Change**: Only affects SlackHandler HTTP requests
- **Backward Compatible**: Existing functionality unchanged
- **Well-Tested Libraries**: requests library is mature and stable
- **Fallback Mechanism**: Automatic detection prevents breaking changes

### Mitigation Strategies

- **Comprehensive Testing**: Unit and integration test coverage
- **Gradual Deployment**: Feature flag for controlled rollout
- **Monitoring**: Real-time success rate tracking
- **Documentation**: Clear troubleshooting guide

## Success Criteria

### Primary Goals

- ✅ Timer expiry notifications reach users successfully
- ✅ No HTTP request hangs or timeouts
- ✅ Existing timer confirmation functionality unchanged
- ✅ Thread-safe operation across all contexts

### Performance Targets

- **Success Rate**: 100% for timer expiry notifications
- **Response Time**: < 5 seconds for HTTP requests
- **Reliability**: No hanging requests or silent failures
- **Compatibility**: Works in all thread contexts

### Monitoring Metrics

- HTTP request completion rate by thread context
- Timer notification delivery success rate
- Response time distribution
- Error rate and types

## Future Enhancements

### Phase 2 Improvements

- **Shared Session Pool**: Better performance with connection reuse
- **Connection Pooling**: Reduce overhead for multiple requests
- **Advanced Error Handling**: Retry logic and circuit breakers

### Phase 3 Advanced Features

- **Event Loop Bridging**: Seamless cross-thread async operations
- **Performance Optimization**: Adaptive threading based on load
- **Enhanced Monitoring**: Detailed metrics and alerting

## Conclusion

This design provides a comprehensive solution to the timer notification threading issue with minimal risk and maximum compatibility. The phased approach allows for quick resolution of the critical issue while laying groundwork for long-term improvements.

The root cause is clearly identified as aiohttp session creation in wrong thread contexts, and the solution provides both immediate fixes and long-term architectural improvements.
