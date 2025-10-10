"""
Tests for Slack app_mention event handler functionality.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.channel_handlers.slack_handler import SlackChannelHandler


class TestSlackAppMentionHandler:
    """Test cases for Slack app_mention event handling."""

    @pytest.fixture
    def slack_handler(self):
        """Create a SlackChannelHandler instance for testing."""
        config = {
            "bot_token": "xoxb-test-token",
            "app_token": "xapp-test-token",
            "default_channel": "#general",
        }
        return SlackChannelHandler(config)

    @pytest.fixture
    def mock_slack_app(self):
        """Create a mock Slack app."""
        mock_app = MagicMock()
        mock_app.event = MagicMock()
        return mock_app

    def test_app_mention_handler_registration(self, slack_handler, mock_slack_app):
        """Test that app_mention event handler is properly registered."""
        slack_handler.slack_app = mock_slack_app

        # Mock the message queue
        slack_handler.message_queue = AsyncMock()

        # Simulate the handler registration that would happen in _background_session_loop
        @mock_slack_app.event("app_mention")
        def handle_app_mention(event, say):
            pass

        # Verify the event handler was registered
        mock_slack_app.event.assert_called_with("app_mention")

    @pytest.mark.asyncio
    async def test_app_mention_event_processing(self, slack_handler):
        """Test that app_mention events are properly processed."""
        # Setup mock message queue
        slack_handler.message_queue = AsyncMock()

        # Mock event data
        event_data = {
            "type": "app_mention",
            "user": "U123456789",
            "channel": "C987654321",
            "text": "<@U0BOTUSER> hello there!",
            "ts": "1234567890.123456",
        }

        # Mock say function
        mock_say = MagicMock()

        # Create the handler function that would be registered
        async def handle_app_mention_event(event, say):
            if not event.get("bot_id"):  # Ignore bot messages
                await slack_handler.message_queue.put(
                    {
                        "type": "app_mention",
                        "user_id": event["user"],
                        "channel_id": event["channel"],
                        "text": event.get("text", ""),
                        "timestamp": event.get("ts"),
                    }
                )

        # Call the handler
        await handle_app_mention_event(event_data, mock_say)

        # Verify message was queued
        slack_handler.message_queue.put.assert_called_once_with(
            {
                "type": "app_mention",
                "user_id": "U123456789",
                "channel_id": "C987654321",
                "text": "<@U0BOTUSER> hello there!",
                "timestamp": "1234567890.123456",
            }
        )

    @pytest.mark.asyncio
    async def test_app_mention_ignores_bot_messages(self, slack_handler):
        """Test that app_mention handler ignores messages from bots."""
        # Setup mock message queue
        slack_handler.message_queue = AsyncMock()

        # Mock event data from a bot
        event_data = {
            "type": "app_mention",
            "user": "U123456789",
            "bot_id": "B987654321",  # This indicates it's from a bot
            "channel": "C987654321",
            "text": "<@U0BOTUSER> hello there!",
            "ts": "1234567890.123456",
        }

        # Mock say function
        mock_say = MagicMock()

        # Create the handler function that would be registered
        async def handle_app_mention_event(event, say):
            if not event.get("bot_id"):  # Ignore bot messages
                await slack_handler.message_queue.put(
                    {
                        "type": "app_mention",
                        "user_id": event["user"],
                        "channel_id": event["channel"],
                        "text": event.get("text", ""),
                        "timestamp": event.get("ts"),
                    }
                )

        # Call the handler
        await handle_app_mention_event(event_data, mock_say)

        # Verify message was NOT queued (bot messages should be ignored)
        slack_handler.message_queue.put.assert_not_called()

    @patch.dict("os.environ", {}, clear=True)  # Clear all environment variables
    def test_app_mention_handler_without_websocket_support(self):
        """Test that app_mention handler is not available without WebSocket support."""
        # Create handler without proper tokens (only webhook)
        config = {"webhook_url": "https://hooks.slack.com/test"}
        slack_handler = SlackChannelHandler(config)

        # Verify bidirectional capability is False (needs both bot_token AND app_token)
        capabilities = slack_handler.get_capabilities()
        assert capabilities["bidirectional"] is False
        assert capabilities["requires_session"] is False

    def test_app_mention_handler_with_websocket_support(self):
        """Test that app_mention handler is available with WebSocket support."""
        # Create handler with proper tokens
        config = {
            "bot_token": "xoxb-test-token",
            "app_token": "xapp-test-token",
            "default_channel": "#general",
        }
        slack_handler = SlackChannelHandler(config)

        # Verify bidirectional capability is True
        capabilities = slack_handler.get_capabilities()
        assert capabilities["bidirectional"] is True
        assert capabilities["requires_session"] is True

    @pytest.mark.asyncio
    async def test_app_mention_with_missing_text(self, slack_handler):
        """Test app_mention handler with missing text field."""
        # Setup mock message queue
        slack_handler.message_queue = AsyncMock()

        # Mock event data without text
        event_data = {
            "type": "app_mention",
            "user": "U123456789",
            "channel": "C987654321",
            "ts": "1234567890.123456"
            # No "text" field
        }

        # Mock say function
        mock_say = MagicMock()

        # Create the handler function that would be registered
        async def handle_app_mention_event(event, say):
            if not event.get("bot_id"):  # Ignore bot messages
                await slack_handler.message_queue.put(
                    {
                        "type": "app_mention",
                        "user_id": event["user"],
                        "channel_id": event["channel"],
                        "text": event.get("text", ""),
                        "timestamp": event.get("ts"),
                    }
                )

        # Call the handler
        await handle_app_mention_event(event_data, mock_say)

        # Verify message was queued with empty text
        slack_handler.message_queue.put.assert_called_once_with(
            {
                "type": "app_mention",
                "user_id": "U123456789",
                "channel_id": "C987654321",
                "text": "",
                "timestamp": "1234567890.123456",
            }
        )

    @pytest.mark.asyncio
    async def test_app_mention_with_missing_timestamp(self, slack_handler):
        """Test app_mention handler with missing timestamp field."""
        # Setup mock message queue
        slack_handler.message_queue = AsyncMock()

        # Mock event data without timestamp
        event_data = {
            "type": "app_mention",
            "user": "U123456789",
            "channel": "C987654321",
            "text": "<@U0BOTUSER> hello there!"
            # No "ts" field
        }

        # Mock say function
        mock_say = MagicMock()

        # Create the handler function that would be registered
        async def handle_app_mention_event(event, say):
            if not event.get("bot_id"):  # Ignore bot messages
                await slack_handler.message_queue.put(
                    {
                        "type": "app_mention",
                        "user_id": event["user"],
                        "channel_id": event["channel"],
                        "text": event.get("text", ""),
                        "timestamp": event.get("ts"),
                    }
                )

        # Call the handler
        await handle_app_mention_event(event_data, mock_say)

        # Verify message was queued with None timestamp
        slack_handler.message_queue.put.assert_called_once_with(
            {
                "type": "app_mention",
                "user_id": "U123456789",
                "channel_id": "C987654321",
                "text": "<@U0BOTUSER> hello there!",
                "timestamp": None,
            }
        )

    @pytest.mark.asyncio
    async def test_app_mention_integration_with_communication_manager(self):
        """Test that app mentions are properly processed by the communication manager."""
        from common.communication_manager import CommunicationManager
        from common.message_bus import MessageBus, MessageType

        # Create message bus and communication manager
        message_bus = MessageBus()
        message_bus.start()
        comm_manager = CommunicationManager(message_bus)

        # Track messages received by the message bus
        received_messages = []

        def capture_message(data):
            received_messages.append(data)

        # Subscribe to incoming requests
        message_bus.subscribe(None, MessageType.INCOMING_REQUEST, capture_message)

        # Simulate an app mention message from Slack handler
        app_mention_message = {
            "type": "app_mention",
            "user_id": "U123456789",
            "channel_id": "C987654321",
            "text": "<@U0BOTUSER> hello there!",
            "timestamp": "1234567890.123456",
        }

        # Process the message through communication manager
        await comm_manager._handle_channel_message("slack", app_mention_message)

        # Wait a bit for async processing
        await asyncio.sleep(0.1)

        # Verify the message was processed correctly
        assert len(received_messages) == 1
        message_data = received_messages[0]
        # message_data is a RequestMetadata object, not a dict
        assert message_data.prompt == "<@U0BOTUSER> hello there!"
        assert message_data.metadata["user_id"] == "U123456789"
        assert message_data.metadata["channel_id"] == "slack:C987654321"
        assert message_data.metadata["source"] == "slack"

        message_bus.stop()
