"""
Unit tests for the Communication Manager.

Tests the functionality of the Communication Manager and channel handlers.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.communication_manager import (
    ChannelHandler,
    ChannelType,
    CommunicationManager,
    DeliveryGuarantee,
    MessageFormat,
)


class MockChannelHandler(ChannelHandler):
    """Mock channel handler for testing."""

    channel_type = ChannelType.CONSOLE

    def __init__(self, config=None, should_succeed=True):
        super().__init__(config)
        self.should_succeed = should_succeed
        self.sent_messages = []

    async def _send(self, message, recipient, message_format, metadata):
        self.sent_messages.append(
            {
                "message": message,
                "recipient": recipient,
                "format": message_format,
                "metadata": metadata,
            }
        )

        if self.should_succeed:
            return {"success": True, "channel": self.channel_type.value}
        else:
            return {"success": False, "error": "Mock failure"}


class TestCommunicationManager(unittest.TestCase):
    """Test cases for the Communication Manager."""

    def setUp(self):
        """Set up test fixtures."""
        # Patch the singleton to allow multiple test instances
        with patch("common.communication_manager.CommunicationManager._instance", None):
            self.manager = CommunicationManager.get_instance()

        # Mock the message bus
        self.mock_message_bus = MagicMock()
        self.manager.message_bus = self.mock_message_bus

        # Create mock channel handlers
        self.console_handler = MockChannelHandler()
        self.slack_handler = MockChannelHandler()
        self.slack_handler.channel_type = ChannelType.SLACK
        self.email_handler = MockChannelHandler()
        self.email_handler.channel_type = ChannelType.EMAIL

        # Register mock handlers
        self.manager.channels = {}  # Clear any auto-discovered handlers
        self.manager.register_channel(self.console_handler)
        self.manager.register_channel(self.slack_handler)
        self.manager.register_channel(self.email_handler)

    def test_singleton_pattern(self):
        """Test that the CommunicationManager follows the singleton pattern."""
        manager1 = CommunicationManager.get_instance()
        manager2 = CommunicationManager.get_instance()
        self.assertIs(manager1, manager2)

    def test_register_channel(self):
        """Test registering a channel handler."""
        # Create a new mock handler
        handler = MockChannelHandler()
        handler.channel_type = ChannelType.WHATSAPP

        # Register it
        self.manager.register_channel(handler)

        # Verify it was registered
        self.assertIn(ChannelType.WHATSAPP, self.manager.channels)
        self.assertIs(self.manager.channels[ChannelType.WHATSAPP], handler)

    def test_unregister_channel(self):
        """Test unregistering a channel handler."""
        # Verify the channel exists first
        self.assertIn(ChannelType.SLACK, self.manager.channels)

        # Unregister it
        self.manager.unregister_channel(ChannelType.SLACK)

        # Verify it was removed
        self.assertNotIn(ChannelType.SLACK, self.manager.channels)

    @pytest.mark.asyncio
    async def test_send_notification_success(self):
        """Test sending a notification successfully."""
        result = await self.manager.send_notification(
            message="Test message",
            channel_type=ChannelType.SLACK,
            recipient="#test-channel",
            delivery_guarantee=DeliveryGuarantee.BEST_EFFORT,
        )

        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["channel"], ChannelType.SLACK.value)

        # Verify the message was sent to the correct handler
        self.assertEqual(len(self.slack_handler.sent_messages), 1)
        sent = self.slack_handler.sent_messages[0]
        self.assertEqual(sent["message"], "Test message")
        self.assertEqual(sent["recipient"], "#test-channel")

    @pytest.mark.asyncio
    async def test_send_notification_with_fallback(self):
        """Test sending a notification with fallback when primary channel fails."""
        # Make the Slack handler fail
        self.slack_handler.should_succeed = False

        result = await self.manager.send_notification(
            message="Test message",
            channel_type=ChannelType.SLACK,
            recipient="#test-channel",
            additional_channels=[ChannelType.EMAIL],
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        )

        # Verify the result shows success from the email channel
        self.assertTrue(result["success"])
        self.assertIn("channels_succeeded", result)
        self.assertIn(ChannelType.EMAIL.value, result["channels_succeeded"])

        # Verify both handlers were attempted
        self.assertEqual(len(self.slack_handler.sent_messages), 1)
        self.assertEqual(len(self.email_handler.sent_messages), 1)

    @pytest.mark.asyncio
    async def test_send_notification_all_fail(self):
        """Test sending a notification when all channels fail."""
        # Make all handlers fail
        self.slack_handler.should_succeed = False
        self.email_handler.should_succeed = False
        self.console_handler.should_succeed = False

        result = await self.manager.send_notification(
            message="Test message",
            channel_type=ChannelType.SLACK,
            additional_channels=[ChannelType.EMAIL],
            delivery_guarantee=DeliveryGuarantee.BEST_EFFORT,
        )

        # Verify the result shows failure
        self.assertFalse(result["success"])
        self.assertIn("All notification channels failed", result["error"])

    @pytest.mark.asyncio
    async def test_at_least_once_notification_tries_all_channels(self):
        """Test that AT_LEAST_ONCE notifications try all available channels."""
        # Make the primary and fallback channels fail
        self.slack_handler.should_succeed = False
        self.email_handler.should_succeed = False

        # But leave console handler working
        self.console_handler.should_succeed = True

        result = await self.manager.send_notification(
            message="Critical message",
            channel_type=ChannelType.SLACK,
            additional_channels=[ChannelType.EMAIL],
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        )

        # Verify the result shows success from multiple channels
        self.assertTrue(result["success"])
        self.assertIn("channels_succeeded", result)
        self.assertIn(ChannelType.CONSOLE.value, result["channels_succeeded"])

        # Verify all handlers were attempted
        self.assertEqual(len(self.slack_handler.sent_messages), 1)
        self.assertEqual(len(self.email_handler.sent_messages), 1)
        self.assertEqual(len(self.console_handler.sent_messages), 1)

    def test_legacy_timer_handler_removed(self):
        """Verify legacy timer handler has been removed in favor of intent-based notifications."""
        # Verify the legacy _handle_timer_expired method no longer exists
        assert not hasattr(
            self.manager, "_handle_timer_expired"
        ), "Legacy _handle_timer_expired method should be removed"

        # Timer notifications now handled via intent-based system
        # See tests/test_single_file_timer_role.py for current timer tests

    @pytest.mark.asyncio
    async def test_exactly_once_delivery_guarantee(self):
        """Test that EXACTLY_ONCE delivery guarantee uses reliable channels."""
        # Make the primary channel fail
        self.slack_handler.should_succeed = False

        # Make email succeed
        self.email_handler.should_succeed = True

        result = await self.manager.send_notification(
            message="Important message",
            channel_type=ChannelType.SLACK,
            recipient="#test-channel",
            delivery_guarantee=DeliveryGuarantee.EXACTLY_ONCE,
        )

        # Verify the result shows success from the email handler
        self.assertTrue(result["success"])
        self.assertEqual(result["channel"], ChannelType.EMAIL.value)

        # Verify both handlers were attempted
        self.assertEqual(len(self.slack_handler.sent_messages), 1)
        self.assertEqual(len(self.email_handler.sent_messages), 1)


if __name__ == "__main__":
    unittest.main()
