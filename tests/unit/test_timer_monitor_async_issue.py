#!/usr/bin/env python3
"""
Test to reproduce the specific timer monitor async issue.

This test reproduces the exact scenario where the timer monitor publishes
a TIMER_EXPIRED message synchronously from an async context, causing the
async callback in communication manager to not execute properly.
"""

import asyncio
import logging
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from common.communication_manager import CommunicationManager
from common.message_bus import MessageBus, MessageType
from supervisor.timer_monitor import TimerMonitor


class TestTimerMonitorAsyncIssue(unittest.TestCase):
    """Test timer monitor async callback execution issue."""

    def setUp(self):
        """Set up test environment."""
        self.message_bus = MessageBus()
        self.message_bus.start()

        # Create communication manager and set up subscriptions
        self.communication_manager = CommunicationManager(self.message_bus)
        self.communication_manager._setup_message_subscriptions()

        # Mock channel for testing
        self.mock_channel = MagicMock()
        self.mock_channel.channel_type.value = "slack"
        self.mock_channel.send = AsyncMock(return_value={"success": True})
        self.communication_manager.channels["slack"] = self.mock_channel

        # Track if route_message was called
        self.route_message_called = False
        self.route_message_args = None

    def tearDown(self):
        """Clean up test environment."""
        self.message_bus.stop()

    @patch("common.communication_manager.CommunicationManager.route_message")
    def test_timer_monitor_sync_publish_to_async_handler(self, mock_route_message):
        """Test that sync timer monitor publish triggers async handler properly."""

        async def mock_route_message_impl(message, context):
            """Mock route_message implementation."""
            self.route_message_called = True
            self.route_message_args = (message, context)
            return [{"success": True}]

        mock_route_message.side_effect = mock_route_message_impl

        # Create a timer that matches the format from the logs
        timer_data = {
            "id": "timer_39a8bd2bbd31474781384f109dbde4cf",
            "name": "30s timer",
            "type": "countdown",
            "user_id": "U52L1U8M6",
            "channel_id": "slack:C52L1UK5E",
            "custom_message": "30s timer expired!",
            "notification_config": {},
            "metadata": {},
        }

        # Create timer monitor with mocked timer manager
        mock_timer_manager = AsyncMock()
        timer_monitor = TimerMonitor(self.message_bus, mock_timer_manager)

        # Simulate the exact scenario: sync method called from async context
        def sync_publish_scenario():
            """Simulate sync publish from timer monitor."""
            timer_monitor._publish_timer_expired_event(timer_data)

        # Run in a thread to simulate the timer monitor context
        thread = threading.Thread(target=sync_publish_scenario)
        thread.start()
        thread.join()

        # Wait for async processing
        time.sleep(0.2)

        # Verify the handler was called
        self.assertTrue(
            self.route_message_called, "route_message should have been called"
        )

        if self.route_message_args:
            message, context = self.route_message_args
            self.assertIn("⏰", message)
            self.assertIn("30s timer expired!", message)
            self.assertEqual(context["channel_id"], "slack:C52L1UK5E")
            self.assertEqual(context["user_id"], "U52L1U8M6")

    def test_message_bus_async_callback_from_sync_context(self):
        """Test message bus handling of async callbacks from sync context."""

        callback_executed = False
        callback_message = None

        async def async_callback(message):
            """Test async callback."""
            nonlocal callback_executed, callback_message
            callback_executed = True
            callback_message = message

        # Subscribe async callback
        self.message_bus.subscribe(self, MessageType.TIMER_EXPIRED, async_callback)

        # Publish from sync context (like timer monitor does)
        def sync_publish():
            test_message = {"timer_id": "test_123", "message": "test"}
            self.message_bus.publish(self, MessageType.TIMER_EXPIRED, test_message)

        # Run in thread to simulate sync context
        thread = threading.Thread(target=sync_publish)
        thread.start()
        thread.join()

        # Wait for async processing
        time.sleep(0.2)

        # Verify callback executed
        self.assertTrue(callback_executed, "Async callback should have executed")
        self.assertIsNotNone(callback_message)
        self.assertEqual(callback_message["timer_id"], "test_123")


if __name__ == "__main__":
    # Configure logging for test visibility
    logging.basicConfig(level=logging.DEBUG)

    # Run tests
    unittest.main()
