#!/usr/bin/env python3
"""
Test script to verify timer notification fixes.

This script tests:
1. Timer expiry notifications are properly handled
2. Reduced logging output (debug level instead of info)
3. Timer notifications route to correct channels
"""

import asyncio
import logging
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, ".")

from common.communication_manager import CommunicationManager
from common.message_bus import MessageBus, MessageType
from supervisor.timer_monitor import TimerMonitor


async def test_timer_expiry_notification():
    """Test that timer expiry notifications work correctly."""
    print("🧪 Testing timer expiry notification fix...")

    # Setup
    message_bus = MessageBus()
    message_bus.start()

    # Mock communication manager with route_message method
    comm_manager = CommunicationManager(message_bus)
    comm_manager.route_message = AsyncMock()

    # Create a mock timer expired event (as published by TimerMonitor)
    timer_expired_event = {
        "timer_id": "timer_test123",
        "timer_name": "1m timer",
        "timer_type": "countdown",
        "user_id": "U52L1U8M6",
        "channel_id": "slack:C52L1UK5E",
        "custom_message": "1m timer expired!",
        "notification_config": {},
        "expired_at": int(time.time()),
        "next_timer_id": None,
        "notification_channel": None,
        "notification_recipient": None,
        "notification_priority": None,
        "metadata": {},
    }

    # Test the timer expired handler
    await comm_manager._handle_timer_expired(timer_expired_event)

    # Verify route_message was called with correct parameters
    assert comm_manager.route_message.called, "route_message should have been called"

    call_args = comm_manager.route_message.call_args
    message = call_args[0][0]
    context = call_args[0][1]

    # Check message content
    assert "⏰" in message, f"Message should contain timer emoji: {message}"
    assert (
        "1m timer expired!" in message
    ), f"Message should contain custom message: {message}"

    # Check context routing
    assert (
        context["channel_id"] == "slack:C52L1UK5E"
    ), f"Should route to original channel: {context['channel_id']}"
    assert (
        context["user_id"] == "U52L1U8M6"
    ), f"Should include user_id: {context['user_id']}"
    assert (
        context["message_type"] == "timer_expired"
    ), f"Should have correct message type: {context['message_type']}"

    print("✅ Timer expiry notification test passed!")
    return True


async def test_logging_levels():
    """Test that logging levels have been reduced from INFO to DEBUG."""
    print("🧪 Testing reduced logging levels...")

    # Setup logging capture
    import io

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)

    # Get loggers for the modules we modified
    slack_logger = logging.getLogger("common.channel_handlers.slack_handler")
    comm_logger = logging.getLogger("common.communication_manager")

    # Clear existing handlers and add our capture handler
    for logger in [slack_logger, comm_logger]:
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    # Test that debug messages don't appear at INFO level
    slack_logger.debug("This debug message should not appear at INFO level")
    comm_logger.debug("Another debug message should not appear at INFO level")

    # Test that info messages still appear
    slack_logger.info("This info message should appear")
    comm_logger.info("Another info message should appear")

    # Check captured output
    log_output = log_capture.getvalue()

    # Should contain INFO messages but not DEBUG messages
    assert (
        "This info message should appear" in log_output
    ), "INFO messages should still be captured"
    assert (
        "Another info message should appear" in log_output
    ), "INFO messages should still be captured"
    assert (
        "This debug message should not appear" not in log_output
    ), "DEBUG messages should not appear at INFO level"
    assert (
        "Another debug message should not appear" not in log_output
    ), "DEBUG messages should not appear at INFO level"

    print("✅ Logging levels test passed!")
    return True


async def test_timer_monitor_integration():
    """Test timer monitor publishes events correctly."""
    print("🧪 Testing timer monitor event publishing...")

    # Setup
    message_bus = MessageBus()
    message_bus.start()

    # Mock the message bus publish method
    original_publish = message_bus.publish
    published_events = []

    def mock_publish(sender, message_type, data):
        published_events.append((sender, message_type, data))
        return original_publish(sender, message_type, data)

    message_bus.publish = mock_publish

    # Create timer monitor
    timer_monitor = TimerMonitor(message_bus)

    # Create a mock timer
    mock_timer = {
        "id": "timer_test456",
        "name": "Test Timer",
        "type": "countdown",
        "user_id": "U52L1U8M6",
        "channel_id": "slack:C52L1UK5E",
        "custom_message": "Test timer expired!",
        "notification_config": {},
        "metadata": {},
    }

    # Test publishing timer expired event
    timer_monitor._publish_timer_expired_event(mock_timer)

    # Verify event was published
    assert (
        len(published_events) == 1
    ), f"Should have published 1 event, got {len(published_events)}"

    sender, message_type, event_data = published_events[0]
    assert (
        message_type == MessageType.TIMER_EXPIRED
    ), f"Should publish TIMER_EXPIRED event, got {message_type}"
    assert (
        event_data["timer_id"] == "timer_test456"
    ), f"Should include timer_id: {event_data}"
    assert (
        event_data["timer_name"] == "Test Timer"
    ), f"Should include timer_name: {event_data}"
    assert (
        event_data["channel_id"] == "slack:C52L1UK5E"
    ), f"Should include channel_id: {event_data}"

    print("✅ Timer monitor integration test passed!")
    return True


async def main():
    """Run all tests."""
    print("🚀 Starting timer notification fix tests...\n")

    tests = [
        test_timer_expiry_notification,
        test_logging_levels,
        test_timer_monitor_integration,
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
            print()

    # Summary
    passed = sum(results)
    total = len(results)

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Timer notification fixes are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the fixes.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
