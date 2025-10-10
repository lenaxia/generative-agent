#!/usr/bin/env python3
"""
Test script to verify timer notification fix.

This script tests that timers created from Slack messages properly
route their expiry notifications back to the original Slack channel.
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_timer_notification_routing():
    """Test that timer expiry notifications are routed to the correct Slack channel."""

    # Mock the timer manager and communication manager
    with patch("roles.timer.lifecycle.get_timer_manager") as mock_get_timer_manager:
        # Create a mock timer manager
        mock_timer_manager = AsyncMock()
        mock_timer_manager.create_timer = AsyncMock(return_value="timer_test123")
        mock_get_timer_manager.return_value = mock_timer_manager

        # Import the function we want to test
        from common.task_context import TaskContext
        from common.task_graph import TaskGraph
        from roles.timer.lifecycle import parse_timer_parameters

        # Create a TaskContext with Slack user and channel info
        task_graph = TaskGraph(tasks=[], dependencies=[])
        context = TaskContext(
            task_graph=task_graph,
            context_id="test_context",
            user_id="U52L1U8M6",  # Slack user ID from the logs
            channel_id="slack:C52L1UK5E",  # Slack channel ID from the logs
        )

        # Test parameters for setting a timer
        parameters = {"action": "set", "duration": "1m"}

        # Call the parse_timer_parameters function
        result = await parse_timer_parameters(
            instruction="set a timer for 1min", context=context, parameters=parameters
        )

        # Verify the timer was created with the correct context
        mock_timer_manager.create_timer.assert_called_once()
        call_args = mock_timer_manager.create_timer.call_args

        # Check that user_id and channel_id were passed correctly
        assert call_args.kwargs["user_id"] == "U52L1U8M6"
        assert call_args.kwargs["channel_id"] == "slack:C52L1UK5E"

        # Verify the result contains success
        assert result["execution_result"]["success"] is True
        assert "timer_test123" in result["execution_result"]["message"]

        logger.info("✅ Timer context routing test passed!")


async def test_timer_expiry_notification():
    """Test that timer expiry notifications are sent to the correct channel."""

    from common.message_bus import MessageBus, MessageType
    from supervisor.timer_monitor import TimerMonitor

    # Create a mock message bus
    mock_message_bus = MagicMock()
    mock_message_bus.publish = MagicMock()

    # Create timer monitor
    timer_monitor = TimerMonitor(mock_message_bus)

    # Mock timer data with Slack channel info
    timer_data = {
        "id": "timer_test123",
        "name": "1m timer",
        "type": "countdown",
        "user_id": "U52L1U8M6",
        "channel_id": "slack:C52L1UK5E",
        "custom_message": "1m timer expired!",
        "notification_config": {},
        "metadata": {},
    }

    # Test the timer expiry event publishing
    timer_monitor._publish_timer_expired_event(timer_data)

    # Verify the message was published with correct data
    mock_message_bus.publish.assert_called_once()
    call_args = mock_message_bus.publish.call_args

    # Check message type
    assert call_args[0][1] == MessageType.TIMER_EXPIRED

    # Check event data
    event_data = call_args[0][2]
    assert event_data["timer_id"] == "timer_test123"
    assert event_data["user_id"] == "U52L1U8M6"
    assert event_data["channel_id"] == "slack:C52L1UK5E"
    assert event_data["custom_message"] == "1m timer expired!"

    logger.info("✅ Timer expiry notification test passed!")


async def test_communication_manager_routing():
    """Test that communication manager routes timer expiry to correct channel."""

    from common.communication_manager import CommunicationManager
    from common.message_bus import MessageBus

    # Create mock message bus
    mock_message_bus = MagicMock()

    # Create communication manager
    comm_manager = CommunicationManager(mock_message_bus)

    # Mock timer expiry message
    timer_message = {
        "timer_id": "timer_test123",
        "timer_name": "1m timer",
        "user_id": "U52L1U8M6",
        "channel_id": "slack:C52L1UK5E",
        "custom_message": "1m timer expired!",
    }

    # Mock route_message method
    with patch.object(
        comm_manager, "route_message", new_callable=AsyncMock
    ) as mock_route:
        # Call the timer expired handler
        await comm_manager._handle_timer_expired(timer_message)

        # Verify route_message was called with correct parameters
        mock_route.assert_called_once()
        call_args = mock_route.call_args

        # Check message content
        message = call_args[0][0]
        assert "1m timer expired!" in message

        # Check context
        context = call_args[0][1]
        assert context["channel_id"] == "slack:C52L1UK5E"
        assert context["user_id"] == "U52L1U8M6"

        logger.info("✅ Communication manager routing test passed!")


async def run_all_tests():
    """Run all tests."""
    logger.info("🧪 Running timer notification fix tests...")

    try:
        await test_timer_notification_routing()
        await test_timer_expiry_notification()
        await test_communication_manager_routing()

        logger.info("🎉 All tests passed! Timer notification fix is working correctly.")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
