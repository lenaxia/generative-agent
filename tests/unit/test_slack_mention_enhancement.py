"""
Tests for Slack @ mention enhancement.

This module tests the enhancement that automatically adds @ mentions
to Slack responses when replying to users, improving user experience
and ensuring proper notification delivery.
"""

from unittest.mock import AsyncMock, patch

import pytest

from common.channel_handlers.slack_handler import SlackChannelHandler
from common.communication_manager import MessageFormat


class TestSlackMentionEnhancement:
    """Test @ mention functionality in Slack responses."""

    @pytest.fixture
    def slack_handler(self):
        """Create Slack handler for testing."""
        with patch("slack_bolt.App"):
            handler = SlackChannelHandler()
            handler.bot_token = "xoxb-test-token"
            handler.webhook_url = None  # Force API usage for testing
            return handler

    @pytest.mark.asyncio
    async def test_send_adds_mention_for_user_responses(self, slack_handler):
        """Test that _send automatically adds @ mention when user_id is provided."""
        # Mock the API send method
        with patch.object(
            slack_handler, "_send_via_api", new_callable=AsyncMock
        ) as mock_api_send:
            mock_api_send.return_value = {"success": True}

            # Send message with user_id in metadata
            message = "Currently in Seattle, it's 61°F and rainy."
            recipient = "#general"
            metadata = {
                "user_id": "U123456",
                "channel_id": "C123456",
                "request_id": "test_request",
            }

            await slack_handler._send(
                message=message,
                recipient=recipient,
                message_format=MessageFormat.PLAIN_TEXT,
                metadata=metadata,
            )

            # Verify the API was called with @ mention added
            mock_api_send.assert_called_once()
            call_args = mock_api_send.call_args[0]
            enhanced_message = call_args[0]  # First argument is the message

            # Should contain @ mention at the beginning
            assert enhanced_message.startswith("<@U123456>")
            assert "Currently in Seattle, it's 61°F and rainy." in enhanced_message

    @pytest.mark.asyncio
    async def test_send_without_user_id_no_mention(self, slack_handler):
        """Test that _send doesn't add @ mention when no user_id is provided."""
        # Mock the API send method
        with patch.object(
            slack_handler, "_send_via_api", new_callable=AsyncMock
        ) as mock_api_send:
            mock_api_send.return_value = {"success": True}

            # Send message without user_id in metadata
            message = "System notification message."
            recipient = "#general"
            metadata = {"channel_id": "C123456", "request_id": "test_request"}

            await slack_handler._send(
                message=message,
                recipient=recipient,
                message_format=MessageFormat.PLAIN_TEXT,
                metadata=metadata,
            )

            # Verify the API was called without @ mention
            mock_api_send.assert_called_once()
            call_args = mock_api_send.call_args[0]
            enhanced_message = call_args[0]  # First argument is the message

            # Should be unchanged
            assert enhanced_message == "System notification message."
            assert "<@" not in enhanced_message

    @pytest.mark.asyncio
    async def test_send_preserves_existing_mentions(self, slack_handler):
        """Test that _send preserves existing @ mentions in the message."""
        # Mock the API send method
        with patch.object(
            slack_handler, "_send_via_api", new_callable=AsyncMock
        ) as mock_api_send:
            mock_api_send.return_value = {"success": True}

            # Send message that already contains @ mention
            message = "<@U789012> asked about weather. Currently in Seattle, it's 61°F and rainy."
            recipient = "#general"
            metadata = {
                "user_id": "U123456",
                "channel_id": "C123456",
                "request_id": "test_request",
            }

            await slack_handler._send(
                message=message,
                recipient=recipient,
                message_format=MessageFormat.PLAIN_TEXT,
                metadata=metadata,
            )

            # Verify the API was called with both mentions
            mock_api_send.assert_called_once()
            call_args = mock_api_send.call_args[0]
            enhanced_message = call_args[0]  # First argument is the message

            # Should contain both mentions
            assert enhanced_message.startswith("<@U123456>")
            assert "<@U789012>" in enhanced_message
            assert "Currently in Seattle, it's 61°F and rainy." in enhanced_message

    @pytest.mark.asyncio
    async def test_send_webhook_also_adds_mentions(self, slack_handler):
        """Test that webhook sending also adds @ mentions."""
        # Configure for webhook usage
        slack_handler.webhook_url = "https://hooks.slack.com/test"
        slack_handler.bot_token = None

        # Mock the webhook send method
        with patch.object(
            slack_handler, "_send_via_webhook", new_callable=AsyncMock
        ) as mock_webhook_send:
            mock_webhook_send.return_value = {"success": True}

            # Send message with user_id in metadata
            message = "Timer expired!"
            recipient = "#general"
            metadata = {
                "user_id": "U123456",
                "channel_id": "C123456",
                "request_id": "timer_request",
            }

            await slack_handler._send(
                message=message,
                recipient=recipient,
                message_format=MessageFormat.PLAIN_TEXT,
                metadata=metadata,
            )

            # Verify the webhook was called with @ mention added
            mock_webhook_send.assert_called_once()
            call_args = mock_webhook_send.call_args[0]
            enhanced_message = call_args[0]  # First argument is the message

            # Should contain @ mention
            assert enhanced_message.startswith("<@U123456>")
            assert "Timer expired!" in enhanced_message

    def test_format_message_with_mention(self, slack_handler):
        """Test the helper method for formatting messages with mentions."""
        # Test with user_id
        message = "Hello world!"
        user_id = "U123456"

        formatted = slack_handler._format_message_with_mention(message, user_id)
        assert formatted == "<@U123456> Hello world!"

        # Test without user_id
        formatted_no_user = slack_handler._format_message_with_mention(message, None)
        assert formatted_no_user == "Hello world!"

        # Test with existing mention
        message_with_mention = "<@U789012> asked about this. Here's the answer."
        formatted_existing = slack_handler._format_message_with_mention(
            message_with_mention, user_id
        )
        assert (
            formatted_existing
            == "<@U123456> <@U789012> asked about this. Here's the answer."
        )
