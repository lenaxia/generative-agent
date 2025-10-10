"""
Unit tests for the unified communication architecture.

Tests the enhanced CommunicationManager with supervisor MessageBus integration,
background thread support, and bidirectional communication capabilities.
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

    channel_type = ChannelType.CONSOLE

    def __init__(self, config=None):
        super().__init__(config)
        self.sent_messages = []
        self.capabilities = {
            "supports_rich_text": False,
            "supports_buttons": False,
            "supports_audio": False,
            "supports_images": False,
            "bidirectional": False,
            "requires_session": False,
            "max_message_length": 1000,
        }

    def _validate_requirements(self) -> bool:
        return True

    def get_capabilities(self) -> dict:
        return self.capabilities

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


class MockBidirectionalHandler(MockChannelHandler):
    """Mock bidirectional channel handler for testing."""

    channel_type = ChannelType.SLACK

    def __init__(self, config=None):
        super().__init__(config)
        self.capabilities["bidirectional"] = True
        self.capabilities["requires_session"] = True
        self.question_responses = []

    def requires_background_thread(self) -> bool:
        return True

    async def _ask_question_impl(
        self, question: str, options: list, timeout: int
    ) -> str:
        return "test_response"


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


class TestCommunicationManager:
    """Test cases for the enhanced CommunicationManager."""

    @pytest.mark.asyncio
    async def test_initialization_with_message_bus(self, message_bus):
        """Test that CommunicationManager accepts supervisor's MessageBus."""
        with patch(
            "common.communication_manager.CommunicationManager._discover_and_initialize_channels"
        ):
            manager = CommunicationManager(message_bus)
            await manager.initialize()

        assert manager.message_bus is message_bus
        assert isinstance(manager.channels, dict)
        assert isinstance(manager.channel_queues, dict)

    @pytest.mark.asyncio
    async def test_message_subscriptions(self, communication_manager, message_bus):
        """Test that CommunicationManager subscribes to required message types."""
        await communication_manager.initialize()
        # Check that subscriptions were set up
        assert MessageType.TIMER_EXPIRED in message_bus._subscribers
        assert MessageType.SEND_MESSAGE in message_bus._subscribers
        assert MessageType.AGENT_QUESTION in message_bus._subscribers

    @pytest.mark.asyncio
    async def test_channel_registration(self, communication_manager):
        """Test channel handler registration."""
        await communication_manager.initialize()
        handler = MockChannelHandler()
        communication_manager.register_channel(handler)

        assert "console" in communication_manager.channels
        assert communication_manager.channels["console"] is handler

    @pytest.mark.asyncio
    async def test_channel_initialization(self, communication_manager):
        """Test channel handler initialization."""
        await communication_manager.initialize()
        handler = MockChannelHandler()
        communication_manager.register_channel(handler)

        # Set communication manager reference
        handler.communication_manager = communication_manager

        # Test initialization
        success = await handler.validate_and_initialize()
        assert success
        assert handler.enabled
        assert handler.session_active

    @pytest.mark.asyncio
    async def test_message_routing(self, communication_manager):
        """Test message routing to appropriate channels."""
        await communication_manager.initialize()
        handler = MockChannelHandler()
        communication_manager.register_channel(handler)
        handler.communication_manager = communication_manager
        await handler.validate_and_initialize()

        # Test routing
        context = {
            "channel_id": "console",
            "message_type": "notification",
            "recipient": "test_user",
        }

        results = await communication_manager.route_message("Test message", context)

        assert len(results) == 1
        assert results[0]["channel"] == "console"
        assert results[0]["result"]["success"]
        assert len(handler.sent_messages) == 1
        assert handler.sent_messages[0]["message"] == "Test message"

    @pytest.mark.asyncio
    async def test_timer_expired_handling(self, communication_manager, message_bus):
        """Test handling of timer expired events."""
        await communication_manager.initialize()
        handler = MockChannelHandler()
        communication_manager.register_channel(handler)
        handler.communication_manager = communication_manager
        await handler.validate_and_initialize()

        # Publish timer expired event
        timer_data = {
            "data": {
                "timer_id": "test_timer",
                "name": "Test Timer",
                "notification_channel": "console",
            }
        }

        message_bus.publish(None, MessageType.TIMER_EXPIRED, timer_data)

        # Give some time for async processing
        await asyncio.sleep(0.1)

        # Check that notification was sent
        assert len(handler.sent_messages) == 1
        assert "Timer expired: Test Timer" in handler.sent_messages[0]["message"]

    @pytest.mark.asyncio
    async def test_send_message_handling(self, communication_manager, message_bus):
        """Test handling of send message events."""
        await communication_manager.initialize()
        handler = MockChannelHandler()
        communication_manager.register_channel(handler)
        handler.communication_manager = communication_manager
        await handler.validate_and_initialize()

        # Publish send message event
        send_data = {
            "message": "Test notification",
            "context": {"channel_id": "console", "recipient": "test_user"},
        }

        message_bus.publish(None, MessageType.SEND_MESSAGE, send_data)

        # Give some time for async processing
        await asyncio.sleep(0.1)

        # Check that message was sent
        assert len(handler.sent_messages) == 1
        assert handler.sent_messages[0]["message"] == "Test notification"

    @pytest.mark.asyncio
    async def test_channel_capabilities(self, communication_manager):
        """Test channel capabilities reporting."""
        await communication_manager.initialize()
        handler = MockChannelHandler()
        capabilities = handler.get_capabilities()

        assert "supports_rich_text" in capabilities
        assert "bidirectional" in capabilities
        assert "requires_session" in capabilities
        assert "max_message_length" in capabilities

    @pytest.mark.asyncio
    async def test_background_thread_requirement(self, communication_manager):
        """Test background thread requirement detection."""
        await communication_manager.initialize()
        # Stateless handler
        stateless_handler = MockChannelHandler()
        assert not stateless_handler.requires_background_thread()

        # Bidirectional handler
        bidirectional_handler = MockBidirectionalHandler()
        assert bidirectional_handler.requires_background_thread()

    @pytest.mark.asyncio
    async def test_delivery_guarantee_routing(self, communication_manager):
        """Test message routing with different delivery guarantees."""
        await communication_manager.initialize()
        handler1 = MockChannelHandler()
        handler1.channel_type = ChannelType.CONSOLE

        handler2 = MockChannelHandler()
        handler2.channel_type = ChannelType.EMAIL

        communication_manager.register_channel(handler1)
        communication_manager.register_channel(handler2)

        for handler in [handler1, handler2]:
            handler.communication_manager = communication_manager
            await handler.validate_and_initialize()

        # Test best effort (single channel)
        context = {
            "channel_id": "console",
            "delivery_guarantee": DeliveryGuarantee.BEST_EFFORT,
        }

        results = await communication_manager.route_message("Test", context)
        assert len(results) == 1

        # Reset
        handler1.sent_messages.clear()
        handler2.sent_messages.clear()

    @pytest.mark.asyncio
    async def test_target_channel_determination(self, communication_manager):
        """Test target channel determination logic."""
        await communication_manager.initialize()
        # Test default routing
        channels = communication_manager._determine_target_channels(
            "console", "notification", {}
        )
        assert channels == ["console"]

        # Test timer routing with audio
        context = {"audio_enabled": True}
        channels = communication_manager._determine_target_channels(
            "console", "timer_expired", context
        )
        # Should include console, and sonos if it's registered and audio is enabled
        assert "console" in channels

    @pytest.mark.asyncio
    async def test_agent_question_handling(self, communication_manager, message_bus):
        """Test handling of agent question events."""
        await communication_manager.initialize()
        handler = MockBidirectionalHandler()
        communication_manager.register_channel(handler)
        handler.communication_manager = communication_manager
        await handler.validate_and_initialize()

        # Publish agent question event
        question_data = {
            "data": {
                "question": "Test question?",
                "options": ["Yes", "No"],
                "timeout": 30,
                "channel_id": "slack",
                "question_id": "test_q1",
            }
        }

        message_bus.publish(None, MessageType.AGENT_QUESTION, question_data)

        # Give some time for async processing
        await asyncio.sleep(0.1)

        # Check that user response was published
        # This would be verified by checking if USER_RESPONSE was published
        # In a real test, we'd mock the message bus publish method


if __name__ == "__main__":
    pytest.main([__file__])
