"""
Unit tests for bidirectional communication in the unified communication architecture.

Tests the enhanced SlackChannelHandler with WebSocket support and bidirectional
communication capabilities.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.channel_handlers.slack_handler import SlackChannelHandler
from common.communication_manager import (
    ChannelHandler,
    ChannelType,
    CommunicationManager,
    DeliveryGuarantee,
    MessageFormat,
)
from common.message_bus import MessageBus, MessageType


@pytest.fixture
def message_bus():
    """Create a MessageBus instance for testing."""
    bus = MessageBus()
    bus.start()
    yield bus
    bus.stop()


@pytest.fixture
def communication_manager(message_bus):
    """Create a CommunicationManager instance for testing."""
    with patch(
        "common.communication_manager.CommunicationManager._discover_and_initialize_channels"
    ):
        manager = CommunicationManager(message_bus)
        return manager


class TestBidirectionalCommunication:
    """Test cases for bidirectional communication."""

    def test_slack_handler_capabilities_without_websocket(self):
        """Test Slack handler capabilities when WebSocket is not available."""
        # Clear environment variables and only provide webhook
        with patch.dict("os.environ", {}, clear=True):
            handler = SlackChannelHandler(
                {"webhook_url": "https://hooks.slack.com/test"}
            )
            capabilities = handler.get_capabilities()

            assert capabilities["supports_rich_text"] is True
            assert capabilities["supports_buttons"] is True
            assert capabilities["bidirectional"] is False
            assert capabilities["requires_session"] is False

    def test_slack_handler_capabilities_with_websocket(self):
        """Test Slack handler capabilities when WebSocket is available."""
        with patch.dict(
            "os.environ",
            {
                "SLACK_BOT_TOKEN": "xoxb-test-token",
                "SLACK_APP_TOKEN": "xapp-test-token",
            },
        ):
            handler = SlackChannelHandler()
            capabilities = handler.get_capabilities()

            assert capabilities["supports_rich_text"] is True
            assert capabilities["supports_buttons"] is True
            assert capabilities["bidirectional"] is True
            assert capabilities["requires_session"] is True

    def test_slack_handler_validation_with_tokens(self):
        """Test Slack handler validation with proper tokens."""
        with patch.dict(
            "os.environ",
            {
                "SLACK_BOT_TOKEN": "xoxb-test-token",
                "SLACK_APP_TOKEN": "xapp-test-token",
            },
        ):
            handler = SlackChannelHandler()
            assert handler._validate_requirements() is True

    def test_slack_handler_validation_webhook_only(self):
        """Test Slack handler validation with webhook only."""
        handler = SlackChannelHandler({"webhook_url": "https://hooks.slack.com/test"})
        assert handler._validate_requirements() is True

    def test_slack_handler_validation_no_credentials(self):
        """Test Slack handler validation without credentials."""
        # Clear environment variables
        with patch.dict("os.environ", {}, clear=True):
            handler = SlackChannelHandler()
            assert handler._validate_requirements() is False

    @pytest.mark.asyncio
    async def test_slack_handler_background_thread_requirement(self):
        """Test that Slack handler requires background thread when WebSocket is available."""
        with patch.dict(
            "os.environ",
            {
                "SLACK_BOT_TOKEN": "xoxb-test-token",
                "SLACK_APP_TOKEN": "xapp-test-token",
            },
        ):
            handler = SlackChannelHandler()
            assert handler.requires_background_thread() is True

    @pytest.mark.asyncio
    async def test_slack_handler_no_background_thread_webhook_only(self):
        """Test that Slack handler doesn't require background thread for webhook-only mode."""
        # Clear environment variables and only provide webhook
        with patch.dict("os.environ", {}, clear=True):
            handler = SlackChannelHandler(
                {"webhook_url": "https://hooks.slack.com/test"}
            )
            assert handler.requires_background_thread() is False

    @pytest.mark.asyncio
    async def test_ask_question_without_websocket(self):
        """Test that ask_question raises error without WebSocket support."""
        # Clear environment variables and only provide webhook
        with patch.dict("os.environ", {}, clear=True):
            handler = SlackChannelHandler(
                {"webhook_url": "https://hooks.slack.com/test"}
            )

            with pytest.raises(NotImplementedError):
                await handler.ask_question("Test question?", ["Yes", "No"])

    @pytest.mark.asyncio
    async def test_ask_question_with_websocket_mock(self):
        """Test ask_question with mocked WebSocket support."""
        with patch.dict(
            "os.environ",
            {
                "SLACK_BOT_TOKEN": "xoxb-test-token",
                "SLACK_APP_TOKEN": "xapp-test-token",
            },
        ):
            handler = SlackChannelHandler()

            # Mock the _send_via_api method
            handler._send_via_api = AsyncMock(
                return_value={"success": True, "ts": "1234567890.123"}
            )

            # Mock the pending questions mechanism
            async def mock_ask_question_impl(question, options, timeout):
                # Simulate immediate response
                return "Yes"

            handler._ask_question_impl = mock_ask_question_impl

            response = await handler.ask_question(
                "Test question?", ["Yes", "No"], timeout=30
            )
            assert response == "Yes"

    @pytest.mark.asyncio
    async def test_create_blocks_with_buttons(self):
        """Test creation of Slack blocks with buttons."""
        handler = SlackChannelHandler()

        buttons = [
            {"text": "Yes", "value": "yes", "style": "primary"},
            {"text": "No", "value": "no"},
        ]

        blocks = handler._create_blocks_with_buttons("Test message", buttons)

        assert len(blocks) == 2
        assert blocks[0]["type"] == "section"
        assert blocks[0]["text"]["text"] == "Test message"
        assert blocks[1]["type"] == "actions"
        assert len(blocks[1]["elements"]) == 2
        assert blocks[1]["elements"][0]["text"]["text"] == "Yes"
        assert blocks[1]["elements"][0]["style"] == "primary"
        assert blocks[1]["elements"][1]["text"]["text"] == "No"

    @pytest.mark.asyncio
    async def test_background_session_without_slack_bolt(self):
        """Test background session when slack_bolt is not available."""
        with patch.dict(
            "os.environ",
            {
                "SLACK_BOT_TOKEN": "xoxb-test-token",
                "SLACK_APP_TOKEN": "xapp-test-token",
            },
        ):
            handler = SlackChannelHandler()

            # Mock import error for slack_bolt
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'slack_bolt'"),
            ):
                # This should not raise an exception, just log an error
                await handler._background_session_loop()

    @pytest.mark.asyncio
    async def test_agent_question_flow_integration(
        self, communication_manager, message_bus
    ):
        """Test complete agent question flow integration."""
        await communication_manager.initialize()

        # Create a mock bidirectional handler
        class MockBidirectionalSlackHandler(SlackChannelHandler):
            def __init__(self):
                super().__init__()
                self.bot_token = "xoxb-test-token"
                self.app_token = "xapp-test-token"
                self.sent_messages = []
                self.question_responses = {}

            async def _send_via_api(
                self, message, channel, message_format, buttons, metadata
            ):
                self.sent_messages.append(
                    {
                        "message": message,
                        "channel": channel,
                        "buttons": buttons,
                        "metadata": metadata,
                    }
                )
                return {"success": True, "ts": "1234567890.123"}

            async def _ask_question_impl(self, question, options, timeout):
                # Simulate user clicking "Yes"
                return "Yes"

        handler = MockBidirectionalSlackHandler()
        communication_manager.register_channel(handler)
        handler.communication_manager = communication_manager
        await handler.validate_and_initialize()

        # Test agent question handling
        question_data = {
            "data": {
                "question": "Do you want to continue?",
                "options": ["Yes", "No"],
                "timeout": 30,
                "channel_id": "slack",
                "question_id": "test_q1",
            }
        }

        # Publish agent question event
        message_bus.publish(None, MessageType.AGENT_QUESTION, question_data)

        # Give some time for async processing
        await asyncio.sleep(0.1)

        # Verify that the question was processed
        # In a real implementation, we'd check that USER_RESPONSE was published

    @pytest.mark.asyncio
    async def test_message_routing_to_bidirectional_channel(
        self, communication_manager
    ):
        """Test message routing to bidirectional channels."""
        await communication_manager.initialize()

        # Create mock bidirectional handler
        class MockBidirectionalHandler(ChannelHandler):
            channel_type = ChannelType.SLACK

            def __init__(self):
                super().__init__()
                self.sent_messages = []

            def get_capabilities(self):
                return {
                    "supports_rich_text": True,
                    "supports_buttons": True,
                    "bidirectional": True,
                    "requires_session": True,
                    "max_message_length": 4000,
                }

            def _validate_requirements(self):
                return True

            async def _send(self, message, recipient, message_format, metadata):
                self.sent_messages.append(
                    {
                        "message": message,
                        "recipient": recipient,
                        "format": message_format,
                        "metadata": metadata,
                    }
                )
                return {"success": True}

            async def _ask_question_impl(self, question, options, timeout):
                return "test_response"

        handler = MockBidirectionalHandler()
        communication_manager.register_channel(handler)
        handler.communication_manager = communication_manager
        await handler.validate_and_initialize()

        # Test routing
        context = {
            "channel_id": "slack",
            "message_type": "notification",
            "recipient": "test_user",
        }

        results = await communication_manager.route_message("Test message", context)

        assert len(results) == 1
        assert results[0]["channel"] == "slack"
        assert results[0]["result"]["success"]
        assert len(handler.sent_messages) == 1


if __name__ == "__main__":
    pytest.main([__file__])
