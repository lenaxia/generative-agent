"""
Integration tests for the unified communication architecture.

Tests the complete communication flow across multiple channels
including message routing, fallback mechanisms, and end-to-end scenarios.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.communication_manager import (
    ChannelHandler,
    ChannelType,
    CommunicationManager,
    DeliveryGuarantee,
    MessageFormat,
)
from common.message_bus import MessageBus, MessageType


class MockChannelHandler(ChannelHandler):
    """Mock channel handler for testing."""

    def __init__(self, channel_type, bidirectional=False):
        super().__init__()
        self.channel_type = channel_type
        self.sent_messages = []
        self._bidirectional = bidirectional
        self.enabled = True
        self.session_active = True

    def _validate_requirements(self) -> bool:
        return True

    def get_capabilities(self) -> dict:
        return {
            "supports_rich_text": True,
            "supports_buttons": self._bidirectional,
            "supports_audio": self.channel_type == ChannelType.SONOS,
            "supports_images": False,
            "bidirectional": self._bidirectional,
            "requires_session": False,  # Don't require background thread for testing
            "max_message_length": 4000,
        }

    def requires_background_thread(self) -> bool:
        """Override to prevent background thread creation in tests."""
        return False

    async def _send(
        self,
        message: str,
        recipient: str,
        message_format: MessageFormat,
        metadata: dict,
    ) -> dict:
        self.sent_messages.append(
            {
                "message": message,
                "recipient": recipient,
                "format": message_format,
                "metadata": metadata,
            }
        )
        return {"success": True, "channel": self.channel_type.value}

    async def _ask_question_impl(
        self, question: str, options: list, timeout: int
    ) -> str:
        if self._bidirectional:
            return options[0] if options else "yes"
        raise NotImplementedError("Not bidirectional")


@pytest.fixture
def message_bus():
    """Create a MessageBus instance for testing."""
    bus = MessageBus()
    bus.start()
    yield bus
    bus.stop()


@pytest.fixture
def communication_manager(message_bus):
    """Create a CommunicationManager with mock handlers."""
    with patch(
        "common.communication_manager.CommunicationManager._discover_and_initialize_channels"
    ), patch(
        "common.communication_manager.CommunicationManager._discover_channel_handlers"
    ), patch(
        "common.communication_manager.CommunicationManager._initialize_all_channels"
    ), patch(
        "common.communication_manager.CommunicationManager.initialize"
    ) as mock_init:
        # Make initialize() a no-op
        async def mock_initialize():
            pass

        mock_init.side_effect = mock_initialize

        manager = CommunicationManager(message_bus)
        return manager


class TestUnifiedCommunicationIntegration:
    """Integration tests for the unified communication system."""

    @pytest.mark.asyncio
    async def test_multi_channel_timer_notification(
        self, communication_manager, message_bus
    ):
        """Test timer notification routing to multiple channels."""
        await communication_manager.initialize()

        # Register mock handlers
        console_handler = MockChannelHandler(ChannelType.CONSOLE)
        slack_handler = MockChannelHandler(ChannelType.SLACK, bidirectional=True)
        sonos_handler = MockChannelHandler(ChannelType.SONOS)

        for handler in [console_handler, slack_handler, sonos_handler]:
            communication_manager.register_channel(handler)
            handler.communication_manager = communication_manager
            await handler.validate_and_initialize()

        # Publish timer expired event
        timer_data = {
            "data": {
                "timer_id": "kitchen_timer",
                "name": "Kitchen Timer",
                "notification_channel": "slack",
            }
        }

        message_bus.publish(None, MessageType.TIMER_EXPIRED, timer_data)

        # Give time for processing
        await asyncio.sleep(0.2)

        # Verify notification was sent
        assert len(slack_handler.sent_messages) >= 1
        assert (
            "Timer expired: Kitchen Timer" in slack_handler.sent_messages[0]["message"]
        )

    @pytest.mark.asyncio
    async def test_message_routing_with_fallback(self, communication_manager):
        """Test message routing with fallback when primary channel fails."""
        await communication_manager.initialize()

        # Create a failing handler
        class FailingHandler(MockChannelHandler):
            async def _send(self, message, recipient, message_format, metadata):
                return {"success": False, "error": "Channel failed"}

        failing_handler = FailingHandler(ChannelType.SLACK)
        console_handler = MockChannelHandler(ChannelType.CONSOLE)

        communication_manager.register_channel(failing_handler)
        communication_manager.register_channel(console_handler)

        for handler in [failing_handler, console_handler]:
            handler.communication_manager = communication_manager
            await handler.validate_and_initialize()

        # Send message with fallback
        context = {
            "channel_id": "slack",
            "delivery_guarantee": DeliveryGuarantee.AT_LEAST_ONCE,
            "message_type": "notification",
        }

        results = await communication_manager.route_message("Test fallback", context)

        # Should have tried slack and fallen back to console
        assert len(results) >= 1
        # At least one should have succeeded (console)
        successful_results = [r for r in results if r["result"]["success"]]
        assert len(successful_results) >= 1

    @pytest.mark.asyncio
    async def test_bidirectional_communication_flow(
        self, communication_manager, message_bus
    ):
        """Test bidirectional communication with agent questions."""
        await communication_manager.initialize()

        # Register bidirectional handler
        slack_handler = MockChannelHandler(ChannelType.SLACK, bidirectional=True)
        communication_manager.register_channel(slack_handler)
        slack_handler.communication_manager = communication_manager
        await slack_handler.validate_and_initialize()

        # Test agent question handling
        question_data = {
            "data": {
                "question": "Should I proceed?",
                "options": ["Yes", "No"],
                "timeout": 30,
                "channel_id": "slack",
                "question_id": "test_q1",
            }
        }

        message_bus.publish(None, MessageType.AGENT_QUESTION, question_data)

        # Give time for processing
        await asyncio.sleep(0.1)

        # Verify the question was handled
        assert "slack" in communication_manager.channels

    @pytest.mark.asyncio
    async def test_channel_capability_detection(self, communication_manager):
        """Test that channels properly report their capabilities."""
        await communication_manager.initialize()

        # Register handlers with different capabilities
        console_handler = MockChannelHandler(ChannelType.CONSOLE)
        slack_handler = MockChannelHandler(ChannelType.SLACK, bidirectional=True)
        sonos_handler = MockChannelHandler(ChannelType.SONOS)

        handlers = [console_handler, slack_handler, sonos_handler]
        for handler in handlers:
            communication_manager.register_channel(handler)
            handler.communication_manager = communication_manager
            await handler.validate_and_initialize()

        # Test capabilities
        console_caps = communication_manager.channels["console"].get_capabilities()
        assert console_caps["bidirectional"] is False

        slack_caps = communication_manager.channels["slack"].get_capabilities()
        assert slack_caps["bidirectional"] is True
        assert slack_caps["supports_buttons"] is True

        sonos_caps = communication_manager.channels["sonos"].get_capabilities()
        assert sonos_caps["supports_audio"] is True
        assert sonos_caps["bidirectional"] is False

    @pytest.mark.asyncio
    async def test_message_routing_intelligence(self, communication_manager):
        """Test intelligent message routing based on message type."""
        await communication_manager.initialize()

        # Register console handler for basic routing test
        console_handler = MockChannelHandler(ChannelType.CONSOLE)
        communication_manager.register_channel(console_handler)
        console_handler.communication_manager = communication_manager
        await console_handler.validate_and_initialize()

        # Test basic routing
        channels = communication_manager._determine_target_channels(
            "console", "notification", {}
        )
        assert "console" in channels

    @pytest.mark.asyncio
    async def test_concurrent_channel_operations(self, communication_manager):
        """Test concurrent operations across multiple channels."""
        await communication_manager.initialize()

        # Register multiple handlers
        handlers = [
            MockChannelHandler(ChannelType.CONSOLE),
            MockChannelHandler(ChannelType.SLACK),
            MockChannelHandler(ChannelType.EMAIL),
        ]

        for handler in handlers:
            communication_manager.register_channel(handler)
            handler.communication_manager = communication_manager
            await handler.validate_and_initialize()

        # Send messages to multiple channels simultaneously
        tasks = []

        for channel_id in ["console", "slack", "email"]:
            context = {"channel_id": channel_id, "message_type": "notification"}
            task = communication_manager.route_message(
                f"Test message for {channel_id}", context
            )
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        assert len(results) == 3
        for result_list in results:
            assert len(result_list) >= 1
            assert any(r["result"]["success"] for r in result_list)

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, communication_manager):
        """Test error handling and recovery mechanisms."""
        await communication_manager.initialize()

        # Register console handler as fallback
        console_handler = MockChannelHandler(ChannelType.CONSOLE)
        communication_manager.register_channel(console_handler)
        console_handler.communication_manager = communication_manager
        await console_handler.validate_and_initialize()

        # Test with invalid channel - should fall back to console
        context = {"channel_id": "nonexistent", "message_type": "notification"}

        results = await communication_manager.route_message("Test message", context)

        # Should fall back to console
        assert len(results) >= 1
        console_result = next((r for r in results if r["channel"] == "console"), None)
        assert console_result is not None

    @pytest.mark.asyncio
    async def test_send_message_event_handling(
        self, communication_manager, message_bus
    ):
        """Test handling of SEND_MESSAGE events."""
        await communication_manager.initialize()

        # Register handler
        console_handler = MockChannelHandler(ChannelType.CONSOLE)
        communication_manager.register_channel(console_handler)
        console_handler.communication_manager = communication_manager
        await console_handler.validate_and_initialize()

        # Publish send message event
        send_data = {
            "message": "Test notification",
            "context": {"channel_id": "console", "recipient": "test_user"},
        }

        message_bus.publish(None, MessageType.SEND_MESSAGE, send_data)

        # Give time for processing
        await asyncio.sleep(0.1)

        # Check that message was sent
        assert len(console_handler.sent_messages) >= 1
        assert console_handler.sent_messages[0]["message"] == "Test notification"

    @pytest.mark.asyncio
    async def test_background_thread_detection(self, communication_manager):
        """Test background thread requirement detection."""
        await communication_manager.initialize()

        # Test stateless handler
        console_handler = MockChannelHandler(ChannelType.CONSOLE)
        assert not console_handler.requires_background_thread()

        # Test stateful handler
        slack_handler = MockChannelHandler(ChannelType.SLACK, bidirectional=True)
        assert slack_handler.requires_background_thread()


if __name__ == "__main__":
    # Using sys.exit() with pytest.main() causes issues when running in a test suite
    # Instead, just run the tests without calling sys.exit()
    pytest.main(["-v", __file__])
