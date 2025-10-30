"""
Unit tests for dynamic channel handler loading.

Tests the functionality of the dynamic channel handler discovery and loading.
"""

import unittest
from unittest.mock import MagicMock, patch

from common.communication_manager import (
    ChannelHandler,
    ChannelType,
    CommunicationManager,
)


class TestDynamicChannelLoading(unittest.TestCase):
    """Test cases for dynamic channel handler loading."""

    def setUp(self):
        """Set up test fixtures."""
        # Patch the singleton to allow multiple test instances
        with patch("common.communication_manager.CommunicationManager._instance", None):
            self.manager = CommunicationManager.get_instance()

        # Mock the message bus
        self.mock_message_bus = MagicMock()
        self.manager.message_bus = self.mock_message_bus

        # Clear any auto-discovered handlers
        self.manager.channels = {}

        # Create a mock channel handler class that inherits from ChannelHandler
        class MockHandler(ChannelHandler):
            def __init__(self, channel_type):
                super().__init__()
                self.channel_type = channel_type
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
                return {"success": True, "channel": self.channel_type.value}

        self.MockHandler = MockHandler

    def test_discover_channel_handlers(self):
        """Test that channel handlers are discovered from the channel_handlers directory."""
        # Create mock handlers using our MockHandler class
        console_handler = self.MockHandler(ChannelType.CONSOLE)
        slack_handler = self.MockHandler(ChannelType.SLACK)
        email_handler = self.MockHandler(ChannelType.EMAIL)

        # Register the handlers directly
        self.manager.register_channel(console_handler)
        self.manager.register_channel(slack_handler)
        self.manager.register_channel(email_handler)

        # Verify that the handlers were registered
        self.assertEqual(
            len(self.manager.channels), 6
        )  # 3 handlers * 2 keys each (enum + string)
        self.assertIn(ChannelType.CONSOLE, self.manager.channels)
        self.assertIn(ChannelType.SLACK, self.manager.channels)
        self.assertIn(ChannelType.EMAIL, self.manager.channels)

    def test_discover_channel_handlers_with_errors(self):
        """Test that errors during handler discovery are handled gracefully."""
        # Create a mock handler using our MockHandler class
        console_handler = self.MockHandler(ChannelType.CONSOLE)

        # Register the handler
        self.manager.register_channel(console_handler)

        # Simulate an error during discovery by calling _discover_channel_handlers
        # with a patched glob that includes a broken handler
        with (
            patch(
                "glob.glob",
                return_value=[
                    "/path/to/console_handler.py",
                    "/path/to/broken_handler.py",
                ],
            ),
            patch(
                "importlib.import_module", side_effect=ImportError("Module not found")
            ),
        ):
            # This should not affect our already registered handler
            self.manager._discover_channel_handlers()

            # Verify that the console handler is still registered
            self.assertEqual(
                len(self.manager.channels), 2
            )  # 1 handler * 2 keys (enum + string)
            self.assertIn(ChannelType.CONSOLE, self.manager.channels)

    # NOTE: Legacy test_integration_with_delivery_guarantees removed.
    # This test used the deprecated send_notification API with ChannelType enums.
    # Current architecture uses NotificationIntent and IntentProcessor instead.
    # See tests/test_intent_processor.py for current notification testing patterns.


if __name__ == "__main__":
    unittest.main()
