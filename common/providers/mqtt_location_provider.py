"""
MQTT-based location provider implementation.

This module provides an MQTT-based implementation of the LocationProvider interface
for Home Assistant integration via MQTT messaging.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Optional

from common.interfaces.context_interfaces import LocationData, LocationProvider
from roles.shared_tools.redis_tools import redis_read, redis_write

logger = logging.getLogger(__name__)


class MQTTLocationProvider(LocationProvider):
    """MQTT-based location tracking via Home Assistant.

    This provider implements the LocationProvider interface using MQTT
    for real-time location updates from Home Assistant and Redis for
    location storage and retrieval.
    """

    def __init__(
        self,
        broker_host: str,
        broker_port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initialize MQTT location provider.

        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port (default: 1883)
            username: MQTT username (optional)
            password: MQTT password (optional)
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self.mqtt_client = None

    async def initialize(self):
        """Setup MQTT client and subscriptions.

        Gracefully disables if MQTT is not configured properly.
        """
        try:
            # Check if MQTT configuration is valid
            if not self.broker_host:
                logger.info(
                    "MQTT location provider disabled - no broker host configured"
                )
                return
            import aiomqtt

            self.mqtt_client = aiomqtt.Client(
                hostname=self.broker_host,
                port=self.broker_port,
                username=self.username,
                password=self.password,
            )

            await self.mqtt_client.connect()
            await self.mqtt_client.subscribe("homeassistant/person/+/state")

            # Start message processing task
            asyncio.create_task(self._process_mqtt_messages())
            logger.info(
                f"MQTT location provider initialized: {self.broker_host}:{self.broker_port}"
            )

        except Exception as e:
            logger.error(f"MQTT location provider initialization failed: {e}")
            raise

    async def _process_mqtt_messages(self):
        """Process MQTT location updates continuously."""
        try:
            async for message in self.mqtt_client.messages:
                await self._process_mqtt_messages_single(message)

        except Exception as e:
            logger.error(f"MQTT message loop failed: {e}")

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

    async def get_current_location(self, user_id: str) -> Optional[str]:
        """Get user's current location from Redis.

        Args:
            user_id: User identifier

        Returns:
            Optional[str]: Current location or None if not available
        """
        try:
            result = redis_read(f"location:{user_id}")
            return result.get("value") if result.get("success") else None
        except Exception as e:
            logger.warning(f"Failed to get location for {user_id}: {e}")
            return None

    async def update_location(
        self, user_id: str, location: str, confidence: float = 1.0
    ) -> bool:
        """Update user location in Redis.

        Args:
            user_id: User identifier
            location: New location string
            confidence: Confidence level (0.0-1.0, not used in current implementation)

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            result = redis_write(f"location:{user_id}", location, ttl=86400)  # 24 hours
            if result.get("success"):
                logger.debug(f"Location updated: {user_id} -> {location}")
                return True
            return False

        except Exception as e:
            logger.warning(f"Failed to update location for {user_id}: {e}")
            return False

    async def get_location_history(
        self, user_id: str, hours: int = 24
    ) -> list[LocationData]:
        """Get location history (placeholder implementation).

        Args:
            user_id: User identifier
            hours: Number of hours of history to retrieve

        Returns:
            List[LocationData]: Location history (empty in placeholder implementation)
        """
        # Placeholder implementation - could be implemented with timestamped Redis keys
        return []
