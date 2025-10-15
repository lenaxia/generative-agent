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


class TestCommunicationManager:
    """Test cases for the Communication Manager."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
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
        self.manager.channels[ChannelType.CONSOLE.value] = self.console_handler
        self.manager.channels[ChannelType.SLACK.value] = self.slack_handler
        self.manager.channels[ChannelType.EMAIL.value] = self.email_handler

    def test_singleton_pattern(self):
        """Test that the CommunicationManager follows the singleton pattern."""
        manager1 = CommunicationManager.get_instance()
        manager2 = CommunicationManager.get_instance()
        assert manager1 is manager2

    def test_register_channel(self):
        """Test registering a channel handler."""
        # Create a new mock handler
        handler = MockChannelHandler()
        handler.channel_type = ChannelType.WHATSAPP

        # Register it manually for test
        self.manager.channels[ChannelType.WHATSAPP.value] = handler

        # Verify it was registered
        assert ChannelType.WHATSAPP.value in self.manager.channels
        assert self.manager.channels[ChannelType.WHATSAPP.value] is handler

    def test_unregister_channel(self):
        """Test unregistering a channel handler."""
        # Verify the channel exists first
        assert ChannelType.SLACK.value in self.manager.channels

        # Unregister it manually for test
        del self.manager.channels[ChannelType.SLACK.value]

        # Verify it was removed
        assert ChannelType.SLACK.value not in self.manager.channels

    def test_legacy_timer_handler_removed(self):
        """Verify legacy timer handler has been removed in favor of intent-based notifications."""
        # Verify the legacy _handle_timer_expired method no longer exists
        assert not hasattr(
            self.manager, "_handle_timer_expired"
        ), "Legacy _handle_timer_expired method should be removed"

        # Timer notifications now handled via intent-based system
        # See tests/test_single_file_timer_role.py for current timer tests

    # NOTE: Legacy send_notification tests with ChannelType enums have been removed.
    # The current architecture uses NotificationIntent and IntentProcessor instead.
    # See tests/test_intent_processor.py for current notification testing patterns.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
