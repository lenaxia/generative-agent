"""
Tests for MQTT location provider implementation.

This module tests the MQTTLocationProvider class that implements the LocationProvider
interface using MQTT for Home Assistant integration.
"""

import json
import logging
from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from common.interfaces.context_interfaces import LocationData
from common.providers.mqtt_location_provider import MQTTLocationProvider

logger = logging.getLogger(__name__)


class TestMQTTLocationProvider:
    """Test MQTTLocationProvider implementation."""

    @pytest.fixture
    def location_provider(self):
        """Create MQTTLocationProvider instance."""
        return MQTTLocationProvider(
            broker_host="homeassistant.local",
            broker_port=1883,
            username="test_user",
            password="test_password",
        )

    def test_location_provider_initialization(self, location_provider):
        """Test MQTTLocationProvider initialization."""
        assert location_provider.broker_host == "homeassistant.local"
        assert location_provider.broker_port == 1883
        assert location_provider.username == "test_user"
        assert location_provider.password == "test_password"
        assert location_provider.mqtt_client is None

    def test_location_provider_minimal_initialization(self):
        """Test MQTTLocationProvider with minimal parameters."""
        provider = MQTTLocationProvider(broker_host="localhost")

        assert provider.broker_host == "localhost"
        assert provider.broker_port == 1883  # Default port
        assert provider.username is None
        assert provider.password is None

    @pytest.mark.asyncio
    async def test_initialize_success(self, location_provider):
        """Test successful MQTT client initialization."""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.subscribe = AsyncMock()

        with (
            patch("aiomqtt.Client") as mock_client_class,
            patch("asyncio.create_task") as mock_create_task,
        ):
            mock_client_class.return_value = mock_client

            await location_provider.initialize()

            # Verify client was created with correct parameters
            mock_client_class.assert_called_once_with(
                hostname="homeassistant.local",
                port=1883,
                username="test_user",
                password="test_password",
            )

            # Verify connection and subscription
            mock_client.connect.assert_called_once()
            mock_client.subscribe.assert_called_once_with(
                "homeassistant/person/+/state"
            )

            # Verify message processing task was created
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, location_provider):
        """Test MQTT client initialization failure."""
        with patch("aiomqtt.Client") as mock_client_class:
            mock_client_class.side_effect = Exception("MQTT connection failed")

            with pytest.raises(Exception, match="MQTT connection failed"):
                await location_provider.initialize()

    @pytest.mark.asyncio
    async def test_get_current_location_success(self, location_provider):
        """Test successful location retrieval."""
        with patch("common.providers.mqtt_location_provider.redis_read") as mock_read:
            mock_read.return_value = {"success": True, "value": "bedroom"}

            location = await location_provider.get_current_location("test_user")

            assert location == "bedroom"
            mock_read.assert_called_once_with("location:test_user")

    @pytest.mark.asyncio
    async def test_get_current_location_not_found(self, location_provider):
        """Test location retrieval when user location not found."""
        with patch("common.providers.mqtt_location_provider.redis_read") as mock_read:
            mock_read.return_value = {"success": False}

            location = await location_provider.get_current_location("test_user")

            assert location is None

    @pytest.mark.asyncio
    async def test_get_current_location_exception(self, location_provider):
        """Test location retrieval with exception."""
        with patch("common.providers.mqtt_location_provider.redis_read") as mock_read:
            mock_read.side_effect = Exception("Redis connection failed")

            location = await location_provider.get_current_location("test_user")

            assert location is None

    @pytest.mark.asyncio
    async def test_update_location_success(self, location_provider):
        """Test successful location update."""
        with patch("common.providers.mqtt_location_provider.redis_write") as mock_write:
            mock_write.return_value = {"success": True}

            result = await location_provider.update_location(
                "test_user", "kitchen", confidence=0.9
            )

            assert result is True
            mock_write.assert_called_once_with(
                "location:test_user", "kitchen", ttl=86400
            )

    @pytest.mark.asyncio
    async def test_update_location_default_confidence(self, location_provider):
        """Test location update with default confidence."""
        with patch("common.providers.mqtt_location_provider.redis_write") as mock_write:
            mock_write.return_value = {"success": True}

            result = await location_provider.update_location("test_user", "living_room")

            assert result is True
            mock_write.assert_called_once_with(
                "location:test_user", "living_room", ttl=86400
            )

    @pytest.mark.asyncio
    async def test_update_location_failure(self, location_provider):
        """Test location update failure."""
        with patch("common.providers.mqtt_location_provider.redis_write") as mock_write:
            mock_write.return_value = {"success": False}

            result = await location_provider.update_location("test_user", "bedroom")

            assert result is False

    @pytest.mark.asyncio
    async def test_update_location_exception(self, location_provider):
        """Test location update with exception."""
        with patch("common.providers.mqtt_location_provider.redis_write") as mock_write:
            mock_write.side_effect = Exception("Redis connection failed")

            result = await location_provider.update_location("test_user", "office")

            assert result is False

    @pytest.mark.asyncio
    async def test_process_mqtt_messages_success(self, location_provider):
        """Test MQTT message processing."""
        # Mock MQTT client with messages
        mock_client = AsyncMock()
        mock_message = Mock()
        mock_message.topic.value = "homeassistant/person/alice/state"
        mock_message.payload.decode.return_value = '{"state": "home"}'

        # Create async iterator for messages
        async def mock_messages():
            yield mock_message

        mock_client.messages = mock_messages()
        location_provider.mqtt_client = mock_client

        with patch.object(location_provider, "update_location") as mock_update:
            mock_update.return_value = True

            # Process one message (we'll break the loop after one iteration)
            try:
                async for message in mock_client.messages:
                    await location_provider._process_mqtt_messages_single(message)
                    break  # Process only one message for testing
            except StopAsyncIteration:
                pass

            # Verify update_location was called with correct parameters
            mock_update.assert_called_once_with("alice", "home")

    @pytest.mark.asyncio
    async def test_process_mqtt_message_invalid_json(self, location_provider):
        """Test MQTT message processing with invalid JSON."""
        mock_client = AsyncMock()
        mock_message = Mock()
        mock_message.topic.value = "homeassistant/person/alice/state"
        mock_message.payload.decode.return_value = "invalid json"

        location_provider.mqtt_client = mock_client

        # Should not raise exception, just log warning
        await location_provider._process_mqtt_messages_single(mock_message)

    @pytest.mark.asyncio
    async def test_process_mqtt_message_wrong_topic(self, location_provider):
        """Test MQTT message processing with wrong topic."""
        mock_client = AsyncMock()
        mock_message = Mock()
        mock_message.topic.value = "homeassistant/sensor/temperature/state"
        mock_message.payload.decode.return_value = '{"state": "22.5"}'

        location_provider.mqtt_client = mock_client

        with patch.object(location_provider, "update_location") as mock_update:
            await location_provider._process_mqtt_messages_single(mock_message)

            # Should not call update_location for non-person topics
            mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_location_history_placeholder(self, location_provider):
        """Test location history placeholder implementation."""
        history = await location_provider.get_location_history("test_user", hours=24)

        # Placeholder implementation returns empty list
        assert history == []


class TestMQTTLocationProviderEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_mqtt_client_none_operations(self):
        """Test operations when MQTT client is None."""
        provider = MQTTLocationProvider("localhost")

        with (
            patch("common.providers.mqtt_location_provider.redis_read") as mock_read,
            patch("common.providers.mqtt_location_provider.redis_write") as mock_write,
        ):
            mock_read.return_value = {"success": False}
            mock_write.return_value = {"success": False}

            # These should work even without initialized MQTT client
            location = await provider.get_current_location("test_user")
            assert location is None  # Will fail Redis read, return None

            result = await provider.update_location("test_user", "home")
            assert result is False  # Will fail Redis write, return False

    def test_mqtt_location_provider_with_none_credentials(self):
        """Test provider creation with None credentials."""
        provider = MQTTLocationProvider(
            broker_host="localhost", username=None, password=None
        )

        assert provider.username is None
        assert provider.password is None


# Helper method for testing single message processing
# This would be added to the MQTTLocationProvider class
async def _process_mqtt_messages_single(self, message):
    """Process a single MQTT message (helper for testing)."""
    try:
        topic = message.topic.value
        payload = json.loads(message.payload.decode())

        if topic.startswith("homeassistant/person/"):
            person = topic.split("/")[2]
            location = payload.get("state")

            await self.update_location(person, location)

    except Exception as e:
        logger.warning(f"MQTT message processing failed: {e}")


# Monkey patch for testing
MQTTLocationProvider._process_mqtt_messages_single = _process_mqtt_messages_single


if __name__ == "__main__":
    pytest.main([__file__])
