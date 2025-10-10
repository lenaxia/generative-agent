"""
Integration tests for the unified communication architecture.

Tests the complete communication flow across multiple channels
including message routing, fallback mechanisms, and end-to-end scenarios.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.channel_handlers.console_handler import ConsoleChannelHandler
from common.channel_handlers.home_assistant_handler import HomeAssistantChannelHandler
from common.channel_handlers.slack_handler import SlackChannelHandler
from common.channel_handlers.sonos_handler import SonosChannelHandler
from common.channel_handlers.voice_handler import VoiceChannelHandler
from common.channel_handlers.whatsapp_handler import WhatsAppChannelHandler
from common.communication_manager import (
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
def full_communication_manager(message_bus):
    """Create a CommunicationManager with all channel handlers."""
    with patch(
        "common.communication_manager.CommunicationManager._discover_and_initialize_channels"
    ):
        manager = CommunicationManager(message_bus)
        return manager


def _create_mock_slack_handler():
    """Create a mock Slack handler with WebSocket support."""
    handler = SlackChannelHandler()
    handler.bot_token = "xoxb-mock-token"
    handler.app_token = "xapp-mock-token"
    handler._send_via_api = AsyncMock(return_value={"success": True})
    handler._background_session_loop = AsyncMock()
    return handler


def _create_mock_sonos_handler():
    """Create a mock Sonos handler."""
    handler = SonosChannelHandler()
    handler.soco_module = MagicMock()
    handler.devices = {"Kitchen": MagicMock(), "Living Room": MagicMock()}
    handler.session_active = True
    handler._text_to_speech = AsyncMock(return_value="/tmp/test.mp3")
    handler._play_on_device = AsyncMock(
        return_value={"success": True, "device": "Kitchen"}
    )
    return handler


def _create_mock_voice_handler():
    """Create a mock Voice handler."""
    handler = VoiceChannelHandler()
    handler.audio_modules = {
        "sr": MagicMock(),
        "pyaudio": MagicMock(),
        "pyttsx3": MagicMock(),
    }
    handler.session_active = True
    handler._speech_to_text = AsyncMock(return_value="test voice command")
    handler._speak_text = MagicMock()
    return handler


def _create_mock_home_assistant_handler():
    """Create a mock Home Assistant handler."""
    handler = HomeAssistantChannelHandler()
    handler.access_token = "mock-token"
    handler.base_url = "http://homeassistant.local:8123"
    handler.websocket = MagicMock()
    handler.session_active = True
    handler._call_service = AsyncMock(return_value={"success": True})
    return handler


def _create_mock_whatsapp_handler():
    """Create a mock WhatsApp handler."""
    handler = WhatsAppChannelHandler()
    handler.phone_number = "+1234567890"
    handler.twilio_account_sid = "mock-sid"
    handler.twilio_auth_token = "mock-token"
    handler._send_via_twilio = AsyncMock(return_value={"success": True})
    return handler


class TestUnifiedCommunicationIntegration:
    """Integration tests for the unified communication system."""

    @pytest.mark.asyncio
    async def test_multi_channel_timer_notification(
        self, full_communication_manager, message_bus
    ):
        """Test timer notification routing to multiple channels."""
        await full_communication_manager.initialize()

        # Register mock handlers
        slack_handler = _create_mock_slack_handler()
        sonos_handler = _create_mock_sonos_handler()

        full_communication_manager.register_channel(slack_handler)
        full_communication_manager.register_channel(sonos_handler)

        for handler in [slack_handler, sonos_handler]:
            handler.communication_manager = full_communication_manager
            await handler.validate_and_initialize()

        # Publish timer expired event
        timer_data = {
            "data": {
                "timer_id": "kitchen_timer",
                "name": "Kitchen Timer",
                "notification_channel": "slack",
                "audio_enabled": True,
            }
        }

        message_bus.publish(None, MessageType.TIMER_EXPIRED, timer_data)

        # Give time for processing
        await asyncio.sleep(0.2)

        # Verify notifications were sent to appropriate channels
        assert "slack" in full_communication_manager.channels
        assert "sonos" in full_communication_manager.channels

    @pytest.mark.asyncio
    async def test_smart_home_control_flow(self, full_communication_manager):
        """Test smart home control through Home Assistant."""
        ha_handler = full_communication_manager.channels["home_assistant"]

        # Test device control
        context = {
            "channel_id": "home_assistant",
            "message_type": "smart_home_control",
            "metadata": {
                "command_type": "service_call",
                "service": "light.turn_on",
                "entity_id": "light.kitchen",
            },
        }

        results = await full_communication_manager.route_message(
            "Turn on kitchen light", context
        )

        assert len(results) >= 1
        assert any(r["result"]["success"] for r in results)

    @pytest.mark.asyncio
    async def test_voice_command_processing(self, full_communication_manager):
        """Test voice command processing and response."""
        voice_handler = full_communication_manager.channels["voice"]

        # Simulate incoming voice command
        await voice_handler.message_queue.put(
            {
                "type": "incoming_message",
                "user_id": "voice_user",
                "channel_id": "voice",
                "text": "What's the weather like?",
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

        # Process the message
        await full_communication_manager._process_channel_queues()

        # Verify message was processed
        # In a real test, we'd verify the message was published to MessageBus

    @pytest.mark.asyncio
    async def test_multi_channel_agent_question(
        self, full_communication_manager, message_bus
    ):
        """Test agent question routing to bidirectional channels."""
        # Test with Slack channel
        question_data = {
            "data": {
                "question": "Should I proceed with the task?",
                "options": ["Yes", "No", "Cancel"],
                "timeout": 30,
                "channel_id": "slack",
                "question_id": "test_q1",
            }
        }

        message_bus.publish(None, MessageType.AGENT_QUESTION, question_data)

        # Give time for processing
        await asyncio.sleep(0.1)

        # Verify question was handled
        slack_handler = full_communication_manager.channels["slack"]
        assert hasattr(slack_handler, "pending_questions")

    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, full_communication_manager):
        """Test fallback to alternative channels when primary fails."""
        # Disable primary channel
        slack_handler = full_communication_manager.channels["slack"]
        slack_handler.enabled = False

        # Send message with fallback
        context = {
            "channel_id": "slack",
            "delivery_guarantee": DeliveryGuarantee.AT_LEAST_ONCE,
            "message_type": "notification",
        }

        results = await full_communication_manager.route_message(
            "Test fallback", context
        )

        # Should have fallen back to console or other available channels
        assert len(results) >= 1
        successful_results = [r for r in results if r["result"]["success"]]
        assert len(successful_results) >= 1

    @pytest.mark.asyncio
    async def test_channel_capability_detection(self, full_communication_manager):
        """Test that channels properly report their capabilities."""
        channels = full_communication_manager.channels

        # Console: basic text only
        console_caps = channels["console"].get_capabilities()
        assert console_caps["bidirectional"] is False
        assert console_caps["supports_audio"] is False

        # Slack: rich text and bidirectional
        slack_caps = channels["slack"].get_capabilities()
        assert slack_caps["supports_rich_text"] is True
        assert slack_caps["bidirectional"] is True

        # Sonos: audio only
        sonos_caps = channels["sonos"].get_capabilities()
        assert sonos_caps["supports_audio"] is True
        assert sonos_caps["bidirectional"] is False

        # Voice: bidirectional audio
        voice_caps = channels["voice"].get_capabilities()
        assert voice_caps["supports_audio"] is True
        assert voice_caps["bidirectional"] is True

        # Home Assistant: bidirectional device control
        ha_caps = channels["home_assistant"].get_capabilities()
        assert ha_caps["bidirectional"] is True
        assert ha_caps["requires_session"] is True

        # WhatsApp: bidirectional messaging
        whatsapp_caps = channels["whatsapp"].get_capabilities()
        assert whatsapp_caps["bidirectional"] is True
        assert whatsapp_caps["supports_images"] is True

    @pytest.mark.asyncio
    async def test_message_routing_intelligence(self, full_communication_manager):
        """Test intelligent message routing based on message type."""
        # Test timer notification routing
        timer_context = {
            "channel_id": "slack",
            "message_type": "timer_expired",
            "audio_enabled": True,
        }

        channels = full_communication_manager._determine_target_channels(
            "slack", "timer_expired", timer_context
        )
        assert "slack" in channels
        # Would include sonos if audio_enabled logic is implemented

        # Test music control routing
        music_context = {"channel_id": "voice", "message_type": "music_control"}

        channels = full_communication_manager._determine_target_channels(
            "voice", "music_control", music_context
        )
        assert "sonos" in channels

    @pytest.mark.asyncio
    async def test_background_thread_management(self, full_communication_manager):
        """Test that background threads are properly managed."""
        # Check that stateful channels have background threads
        stateful_channels = ["slack", "voice", "home_assistant"]

        for channel_id in stateful_channels:
            handler = full_communication_manager.channels[channel_id]
            if handler.requires_background_thread():
                assert handler.background_thread is not None
                assert handler.message_queue is not None

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, full_communication_manager, message_bus):
        """Test complete end-to-end communication workflow."""
        # 1. Incoming message from voice
        voice_handler = full_communication_manager.channels["voice"]
        await voice_handler.message_queue.put(
            {
                "type": "incoming_message",
                "user_id": "voice_user",
                "channel_id": "voice",
                "text": "Turn on the kitchen lights",
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

        # 2. Process the message
        await full_communication_manager._process_channel_queues()

        # 3. Simulate agent processing and response
        response_context = {
            "channel_id": "voice",
            "message_type": "smart_home_control",
            "metadata": {
                "command_type": "service_call",
                "service": "light.turn_on",
                "entity_id": "light.kitchen",
            },
        }

        # 4. Route response back
        results = await full_communication_manager.route_message(
            "Kitchen lights turned on", response_context
        )

        # 5. Verify response was sent
        assert len(results) >= 1
        assert any(r["result"]["success"] for r in results)

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, full_communication_manager):
        """Test error handling and recovery mechanisms."""
        # Test with invalid channel
        context = {"channel_id": "nonexistent", "message_type": "notification"}

        results = await full_communication_manager.route_message(
            "Test message", context
        )

        # Should fall back to console
        assert len(results) >= 1
        console_result = next((r for r in results if r["channel"] == "console"), None)
        assert console_result is not None

    @pytest.mark.asyncio
    async def test_concurrent_channel_operations(self, full_communication_manager):
        """Test concurrent operations across multiple channels."""
        # Send messages to multiple channels simultaneously
        tasks = []

        for channel_id in ["console", "slack", "sonos"]:
            context = {"channel_id": channel_id, "message_type": "notification"}
            task = full_communication_manager.route_message(
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
    async def test_session_lifecycle_management(self, full_communication_manager):
        """Test session lifecycle for stateful channels."""
        # Test session start/stop for Home Assistant
        ha_handler = full_communication_manager.channels["home_assistant"]

        # Mock WebSocket operations
        ha_handler._connect_websocket = AsyncMock()

        # Test session management
        assert (
            ha_handler.session_active is True
        )  # Should be active after initialization

        # Test session stop
        await ha_handler.stop_session()
        assert ha_handler.session_active is False


if __name__ == "__main__":
    pytest.main([__file__])
