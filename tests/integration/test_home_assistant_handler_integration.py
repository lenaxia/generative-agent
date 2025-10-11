"""
Integration tests for the Home Assistant channel handler.

This test suite verifies the integration of the Home Assistant channel handler
with external Home Assistant instances, testing device control, state monitoring,
and bidirectional communication.
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from common.channel_handlers.home_assistant_handler import HomeAssistantChannelHandler
from common.communication_manager import ChannelType, MessageFormat


@pytest.fixture
def mock_websockets():
    """Mock websockets library and connection."""
    with patch.dict("sys.modules", {"websockets": Mock()}):
        import sys

        mock_ws = AsyncMock()
        mock_ws_connect = AsyncMock(return_value=mock_ws)
        sys.modules["websockets"].connect = mock_ws_connect

        # Setup authentication response sequence
        mock_ws.recv = AsyncMock()
        mock_ws.recv.side_effect = [
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            json.dumps(
                {
                    "id": 1,
                    "type": "result",
                    "success": True,
                    "result": [],  # Empty initial subscription result
                }
            ),
        ]

        yield mock_ws


@pytest.fixture
def ha_handler():
    """Create a Home Assistant channel handler for testing."""
    # Override environment variables
    os.environ["HOME_ASSISTANT_URL"] = "http://homeassistant.test:8123"
    os.environ["HOME_ASSISTANT_TOKEN"] = "test_token"

    config = {
        "monitored_domains": ["light", "switch", "sensor"],
        "monitored_entities": ["light.living_room", "sensor.temperature"],
    }

    handler = HomeAssistantChannelHandler(config)
    yield handler

    # Clean up
    if "HOME_ASSISTANT_URL" in os.environ:
        del os.environ["HOME_ASSISTANT_URL"]
    if "HOME_ASSISTANT_TOKEN" in os.environ:
        del os.environ["HOME_ASSISTANT_TOKEN"]


class TestHomeAssistantHandlerIntegration:
    """Integration tests for Home Assistant channel handler."""

    @pytest.mark.asyncio
    @patch("asyncio.Future")
    async def test_websocket_connection_and_auth(
        self, mock_future, mock_websockets, ha_handler
    ):
        """Test websocket connection and authentication flow."""
        # Setup
        message_queue = asyncio.Queue()
        ha_handler.message_queue = message_queue
        ha_handler.session_active = True

        # Run connection in background
        connection_task = asyncio.create_task(ha_handler._background_session_loop())

        # Give the task time to process
        await asyncio.sleep(0.1)

        # Stop the background task
        ha_handler.session_active = False
        await asyncio.sleep(0.1)
        connection_task.cancel()

        try:
            await connection_task
        except asyncio.CancelledError:
            pass

        # Verify connection and authentication
        expected_auth_msg = json.dumps({"type": "auth", "access_token": "test_token"})
        expected_subscribe_msg = json.dumps(
            {"id": 1, "type": "subscribe_events", "event_type": "state_changed"}
        )

        # Check connection was made with auth header
        mock_websockets.connect.assert_called_once()
        assert (
            "ws://homeassistant.test:8123/api/websocket"
            in mock_websockets.connect.call_args[0][0]
        )
        assert (
            mock_websockets.connect.call_args[1]["extra_headers"]["Authorization"]
            == "Bearer test_token"
        )

        # Check authentication and subscription messages
        mock_websockets.send.assert_any_call(expected_auth_msg)
        mock_websockets.send.assert_any_call(expected_subscribe_msg)

    @pytest.mark.asyncio
    @patch("asyncio.Future")
    async def test_handle_state_change_event(
        self, mock_future, mock_websockets, ha_handler
    ):
        """Test handling state change events from Home Assistant."""
        # Setup
        message_queue = asyncio.Queue()
        ha_handler.message_queue = message_queue
        ha_handler.websocket = mock_websockets

        # Create a state change event
        state_change_event = {
            "type": "event",
            "event": {
                "event_type": "state_changed",
                "data": {
                    "entity_id": "light.living_room",
                    "old_state": {
                        "state": "off",
                        "attributes": {"friendly_name": "Living Room Light"},
                    },
                    "new_state": {
                        "state": "on",
                        "attributes": {
                            "friendly_name": "Living Room Light",
                            "brightness": 255,
                        },
                        "last_changed": "2023-01-01T12:00:00Z",
                    },
                },
            },
        }

        # Process the event
        await ha_handler._handle_websocket_message(state_change_event)

        # Verify entity state was updated
        assert "light.living_room" in ha_handler.entity_states
        assert ha_handler.entity_states["light.living_room"]["state"] == "on"

        # Verify notification was sent to message queue
        notification = await message_queue.get()
        assert notification["type"] == "incoming_message"
        assert notification["channel_id"] == "home_assistant"
        assert "Living Room Light changed from off to on" in notification["text"]
        assert notification["metadata"]["entity_id"] == "light.living_room"

    @pytest.mark.asyncio
    async def test_call_service(self, mock_websockets, ha_handler):
        """Test calling a Home Assistant service."""
        # Setup
        ha_handler.websocket = mock_websockets

        # Mock response from Home Assistant
        mock_websockets.send = AsyncMock()

        # Setup response future
        response_future = asyncio.Future()
        response_future.set_result({"success": True, "result": {}})
        ha_handler.pending_commands = {
            2: response_future
        }  # 2 is the expected command ID

        # Execute service call
        result = await ha_handler._call_service(
            message="Turn on the lights",
            recipient="light.living_room",
            metadata={"service": "light.turn_on", "service_data": {"brightness": 255}},
        )

        # Verify service call was made
        assert mock_websockets.send.call_count == 1
        service_call_data = json.loads(mock_websockets.send.call_args[0][0])
        assert service_call_data["type"] == "call_service"
        assert service_call_data["domain"] == "light"
        assert service_call_data["service"] == "turn_on"
        assert service_call_data["service_data"]["brightness"] == 255
        assert service_call_data["target"]["entity_id"] == "light.living_room"

        # Verify result
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_parse_service_from_message(self, ha_handler):
        """Test parsing service name from natural language."""
        # Test various message patterns
        assert (
            ha_handler._parse_service_from_message("Turn on the kitchen light")
            == "homeassistant.turn_on"
        )
        assert (
            ha_handler._parse_service_from_message("Turn off all lights")
            == "homeassistant.turn_off"
        )
        assert (
            ha_handler._parse_service_from_message("Toggle the living room lamp")
            == "homeassistant.toggle"
        )
        assert (
            ha_handler._parse_service_from_message("Set temperature to 72")
            == "climate.set_temperature"
        )
        assert (
            ha_handler._parse_service_from_message("Dim the lights to 50%")
            == "light.turn_on"
        )
        assert ha_handler._parse_service_from_message("Unknown command") is None

    @pytest.mark.asyncio
    @patch("asyncio.Future")
    async def test_get_entity_state(self, mock_future, mock_websockets, ha_handler):
        """Test getting entity state."""
        # Setup
        ha_handler.websocket = mock_websockets
        mock_websockets.send = AsyncMock()

        # Mock state response
        state_response = [
            {
                "entity_id": "light.living_room",
                "state": "on",
                "attributes": {"friendly_name": "Living Room Light", "brightness": 255},
            },
            {
                "entity_id": "sensor.temperature",
                "state": "72",
                "attributes": {
                    "friendly_name": "Temperature",
                    "unit_of_measurement": "°F",
                },
            },
        ]

        # Setup response future
        response_future = asyncio.Future()
        response_future.set_result(state_response)
        ha_handler.pending_commands = {2: response_future}

        # Execute state query
        result = await ha_handler._get_entity_state("light.living_room")

        # Verify request was made
        mock_websockets.send.assert_called_once()
        request_data = json.loads(mock_websockets.send.call_args[0][0])
        assert request_data["type"] == "get_states"

        # Verify result
        assert result["success"] is True
        assert result["state"]["entity_id"] == "light.living_room"
        assert result["state"]["state"] == "on"
        assert result["state"]["attributes"]["brightness"] == 255

    @pytest.mark.asyncio
    async def test_send_impl(self, mock_websockets, ha_handler):
        """Test sending commands to Home Assistant."""
        # Setup
        ha_handler._call_service = AsyncMock(return_value={"success": True})
        ha_handler._get_entity_state = AsyncMock(return_value={"success": True})
        ha_handler._send_notification_to_ha = AsyncMock(return_value={"success": True})

        # Test service call
        result1 = await ha_handler._send(
            message="Turn on lights",
            recipient="light.living_room",
            message_format=MessageFormat.TEXT,
            metadata={
                "command_type": "service_call",
                "service": "light.turn_on",
                "service_data": {"brightness": 255},
            },
        )

        # Test get state
        result2 = await ha_handler._send(
            message="Get temperature",
            recipient="sensor.temperature",
            message_format=MessageFormat.TEXT,
            metadata={"command_type": "get_state"},
        )

        # Test notification
        result3 = await ha_handler._send(
            message="Alert: Motion detected",
            recipient=None,
            message_format=MessageFormat.TEXT,
            metadata={"command_type": "notification", "title": "Security Alert"},
        )

        # Test unknown command type
        result4 = await ha_handler._send(
            message="Unknown command",
            recipient=None,
            message_format=MessageFormat.TEXT,
            metadata={"command_type": "unknown"},
        )

        # Verify calls were made
        ha_handler._call_service.assert_called_once()
        ha_handler._get_entity_state.assert_called_once_with("sensor.temperature")
        ha_handler._send_notification_to_ha.assert_called_once()

        # Verify results
        assert result1["success"] is True
        assert result2["success"] is True
        assert result3["success"] is True
        assert result4["success"] is False
        assert "Unknown command type" in result4["error"]

    @pytest.mark.asyncio
    async def test_control_device(self, ha_handler):
        """Test controlling a Home Assistant device."""
        # Setup
        ha_handler._call_service = AsyncMock(return_value={"success": True})

        # Control device
        result = await ha_handler.control_device(
            entity_id="light.living_room",
            action="turn_on",
            parameters={"brightness": 255},
        )

        # Verify service call
        ha_handler._call_service.assert_called_once()
        call_args = ha_handler._call_service.call_args[0]
        assert "turn_on light.living_room" in call_args[0]
        assert call_args[1] == "light.living_room"
        assert call_args[2]["service"] == "homeassistant.turn_on"
        assert call_args[2]["service_data"]["brightness"] == 255

        # Verify result
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_list_entities(self, mock_websockets, ha_handler):
        """Test listing entities from Home Assistant."""
        # Setup
        ha_handler.websocket = mock_websockets
        mock_websockets.send = AsyncMock()

        # Mock entities response
        entities = [
            {
                "entity_id": "light.living_room",
                "state": "on",
                "attributes": {"friendly_name": "Living Room Light"},
            },
            {
                "entity_id": "light.kitchen",
                "state": "off",
                "attributes": {"friendly_name": "Kitchen Light"},
            },
            {
                "entity_id": "sensor.temperature",
                "state": "72",
                "attributes": {
                    "friendly_name": "Temperature",
                    "unit_of_measurement": "°F",
                },
            },
        ]

        # Setup response future
        response_future = asyncio.Future()
        response_future.set_result(entities)
        ha_handler.pending_commands = {2: response_future}

        # List all entities
        result1 = await ha_handler.list_entities()

        # List entities filtered by domain
        response_future = asyncio.Future()
        response_future.set_result(entities)
        ha_handler.pending_commands = {3: response_future}
        result2 = await ha_handler.list_entities(domain="light")

        # Verify requests
        assert mock_websockets.send.call_count >= 2

        # Verify results
        assert result1["success"] is True
        assert result1["count"] == 3
        assert len(result1["entities"]) == 3

        assert result2["success"] is True
        assert result2["count"] == 2
        assert len(result2["entities"]) == 2
        assert all(e["entity_id"].startswith("light.") for e in result2["entities"])

    @pytest.mark.asyncio
    async def test_should_notify_state_change(self, ha_handler):
        """Test notification filtering logic."""
        # Setup
        ha_handler.monitored_entities = ["light.living_room", "sensor.temperature"]
        ha_handler.monitored_domains = ["light", "switch", "sensor"]

        # Test cases
        # 1. Monitored entity with state change
        assert (
            ha_handler._should_notify_state_change(
                entity_id="light.living_room",
                new_state={"state": "on"},
                old_state={"state": "off"},
            )
            is True
        )

        # 2. Monitored entity with same state (no change)
        assert (
            ha_handler._should_notify_state_change(
                entity_id="light.living_room",
                new_state={"state": "on"},
                old_state={"state": "on"},
            )
            is False
        )

        # 3. Monitored domain but not in entity list
        assert (
            ha_handler._should_notify_state_change(
                entity_id="light.bedroom",
                new_state={"state": "on"},
                old_state={"state": "off"},
            )
            is False
        )

        # 4. Unmonitored domain
        assert (
            ha_handler._should_notify_state_change(
                entity_id="media_player.tv",
                new_state={"state": "playing"},
                old_state={"state": "idle"},
            )
            is False
        )

        # 5. Empty monitored_entities list - should allow all entities in monitored domains
        ha_handler.monitored_entities = []
        assert (
            ha_handler._should_notify_state_change(
                entity_id="light.bedroom",
                new_state={"state": "on"},
                old_state={"state": "off"},
            )
            is True
        )

    def test_format_state_change_message(self, ha_handler):
        """Test formatting state change messages."""
        # Test with friendly name
        msg1 = ha_handler._format_state_change_message(
            entity_id="light.living_room",
            new_state={
                "state": "on",
                "attributes": {"friendly_name": "Living Room Light"},
            },
            old_state={
                "state": "off",
                "attributes": {"friendly_name": "Living Room Light"},
            },
        )
        assert msg1 == "🏠 Living Room Light changed from off to on"

        # Test with entity ID only (no friendly name)
        msg2 = ha_handler._format_state_change_message(
            entity_id="binary_sensor.motion",
            new_state={"state": "on", "attributes": {}},
            old_state={"state": "off", "attributes": {}},
        )
        assert msg2 == "🏠 binary_sensor.motion changed from off to on"

        # Test with no old state (new entity)
        msg3 = ha_handler._format_state_change_message(
            entity_id="sensor.temperature",
            new_state={"state": "72", "attributes": {"friendly_name": "Temperature"}},
            old_state=None,
        )
        assert msg3 == "🏠 Temperature changed from unknown to 72"

    def test_validate_requirements(self, ha_handler):
        """Test requirements validation."""
        # Test with valid configuration
        assert ha_handler._validate_requirements() is True

        # Test with missing token
        os.environ.pop("HOME_ASSISTANT_TOKEN", None)
        assert ha_handler._validate_requirements() is False

        # Test with missing URL
        os.environ["HOME_ASSISTANT_TOKEN"] = "test_token"
        os.environ.pop("HOME_ASSISTANT_URL", None)
        assert ha_handler._validate_requirements() is False

        # Test with missing websockets
        os.environ["HOME_ASSISTANT_URL"] = "http://homeassistant.test:8123"
        with patch.dict("sys.modules", {"websockets": None}):
            assert ha_handler._validate_requirements() is False
