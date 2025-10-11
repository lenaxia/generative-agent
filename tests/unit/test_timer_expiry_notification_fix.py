#!/usr/bin/env python3
"""
Test for timer expiry notification fix.

This test reproduces and validates the fix for the issue where timer expiry
notifications are not delivered due to async callback execution problems.
"""

import asyncio
import logging
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.communication_manager import CommunicationManager
from common.message_bus import MessageBus, MessageType


class TestTimerExpiryNotificationFix(unittest.TestCase):
    """Test timer expiry notification delivery fix."""

    def setUp(self):
        """Set up test environment."""
        self.message_bus = MessageBus()
        self.message_bus.start()
        self.communication_manager = CommunicationManager(self.message_bus)

        # Mock channel for testing
        self.mock_channel = MagicMock()
        self.mock_channel.channel_type.value = "slack"
        self.mock_channel.send = AsyncMock(return_value={"success": True})
        self.communication_manager.channels["slack"] = self.mock_channel

        # Track callback executions
        self.callback_executed = False
        self.callback_error = None

    def tearDown(self):
        """Clean up test environment."""
        self.message_bus.stop()

    def test_async_callback_execution_with_event_loop(self):
        """Test that async callbacks execute properly when event loop exists."""

        async def test_callback(message):
            """Test async callback."""
            self.callback_executed = True
            self.assertEqual(message["timer_id"], "test_timer")

        # Subscribe to timer expired events
        self.message_bus.subscribe(self, MessageType.TIMER_EXPIRED, test_callback)

        async def run_test():
            """Run the test in an async context."""
            # Publish timer expired message
            timer_message = {
                "timer_id": "test_timer",
                "timer_name": "Test Timer",
                "user_id": "test_user",
                "channel_id": "slack:test_channel",
                "custom_message": "Test timer expired!",
            }

            self.message_bus.publish(self, MessageType.TIMER_EXPIRED, timer_message)

            # Wait for async callback to execute
            await asyncio.sleep(0.1)

            self.assertTrue(
                self.callback_executed, "Async callback should have executed"
            )

        # Run test in event loop
        asyncio.run(run_test())

    def test_async_callback_execution_without_event_loop(self):
        """Test that async callbacks execute properly when no event loop exists."""

        async def test_callback(message):
            """Test async callback."""
            self.callback_executed = True
            self.assertEqual(message["timer_id"], "test_timer")

        # Subscribe to timer expired events
        self.message_bus.subscribe(self, MessageType.TIMER_EXPIRED, test_callback)

        # Publish timer expired message from thread without event loop
        def publish_from_thread():
            timer_message = {
                "timer_id": "test_timer",
                "timer_name": "Test Timer",
                "user_id": "test_user",
                "channel_id": "slack:test_channel",
                "custom_message": "Test timer expired!",
            }

            self.message_bus.publish(self, MessageType.TIMER_EXPIRED, timer_message)

        # Run in separate thread (no event loop)
        thread = threading.Thread(target=publish_from_thread)
        thread.start()
        thread.join()

        # Wait for callback to execute
        time.sleep(0.2)

        self.assertTrue(
            self.callback_executed,
            "Async callback should have executed in new event loop",
        )

    @patch("common.communication_manager.CommunicationManager.route_message")
    async def test_timer_expired_handler_execution(self, mock_route_message):
        """Test that _handle_timer_expired is called and executes properly."""
        mock_route_message.return_value = [{"success": True}]

        # Initialize communication manager (sets up subscriptions)
        self.communication_manager._setup_message_subscriptions()

        # Publish timer expired message
        timer_message = {
            "timer_id": "test_timer_123",
            "timer_name": "Test Timer",
            "user_id": "U12345",
            "channel_id": "slack:C12345",
            "custom_message": "Test timer expired!",
            "notification_config": {},
        }

        self.message_bus.publish(self, MessageType.TIMER_EXPIRED, timer_message)

        # Wait for async processing
        await asyncio.sleep(0.1)

        # Verify route_message was called
        mock_route_message.assert_called_once()

        # Verify the call arguments
        call_args = mock_route_message.call_args
        message_arg = call_args[0][0]
        context_arg = call_args[0][1]

        self.assertIn("⏰", message_arg)
        self.assertIn("Test timer expired!", message_arg)
        self.assertEqual(context_arg["channel_id"], "slack:C12345")
        self.assertEqual(context_arg["user_id"], "U12345")
        self.assertEqual(context_arg["message_type"], "timer_expired")

    def test_communication_manager_subscription_setup(self):
        """Test that communication manager properly subscribes to timer events."""
        # Check that timer expired subscription exists
        timer_subscribers = self.message_bus._subscribers.get(
            MessageType.TIMER_EXPIRED, {}
        )

        # Communication manager should be subscribed
        self.assertIn(self.communication_manager, timer_subscribers)

        # Should have the _handle_timer_expired callback
        callbacks = timer_subscribers[self.communication_manager]
        self.assertEqual(len(callbacks), 1)

        # Verify it's the right callback
        callback = callbacks[0]
        self.assertEqual(callback.__name__, "_handle_timer_expired")


if __name__ == "__main__":
    # Configure logging for test visibility
    logging.basicConfig(level=logging.DEBUG)

    # Run tests
    unittest.main()
