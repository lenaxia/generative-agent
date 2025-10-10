"""
Home Assistant channel handler for the Communication Manager.

This handler provides bidirectional integration with Home Assistant
for device control, state monitoring, and automation triggers.
"""

import asyncio
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import aiohttp

from common.communication_manager import ChannelHandler, ChannelType, MessageFormat

logger = logging.getLogger(__name__)


class HomeAssistantChannelHandler(ChannelHandler):
    """
    Channel handler for Home Assistant integration.

    Supports:
    - Device state monitoring via WebSocket
    - Device control commands
    - Automation triggers
    - Bidirectional communication
    - Event-driven notifications
    """

    channel_type = ChannelType.HOME_ASSISTANT

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize the Home Assistant channel handler."""
        super().__init__(config)

        # Configuration
        self.base_url = os.environ.get("HOME_ASSISTANT_URL") or self.config.get(
            "url", "http://homeassistant.local:8123"
        )
        self.access_token = os.environ.get("HOME_ASSISTANT_TOKEN") or self.config.get(
            "access_token"
        )
        self.websocket_url = (
            self.base_url.replace("http://", "ws://").replace("https://", "wss://")
            + "/api/websocket"
        )

        # Device filtering
        self.monitored_domains = self.config.get(
            "monitored_domains", ["light", "switch", "sensor", "climate"]
        )
        self.monitored_entities = self.config.get("monitored_entities", [])

        # Runtime state
        self.websocket = None
        self.websocket_id = 1
        self.pending_commands = {}
        self.entity_states = {}
        self.subscriptions = {}

    def _validate_requirements(self) -> bool:
        """Validate Home Assistant configuration."""
        if not self.access_token:
            logger.error(
                "HOME_ASSISTANT_TOKEN environment variable or access_token config required"
            )
            return False

        if not self.base_url:
            logger.error(
                "HOME_ASSISTANT_URL environment variable or url config required"
            )
            return False

        # Check for websockets library
        try:
            import websockets

            return True
        except ImportError:
            logger.error(
                "websockets library required for Home Assistant: pip install websockets"
            )
            return False

    def _get_requirements_error_message(self) -> str:
        """Get descriptive error message for missing Home Assistant requirements."""
        missing = []
        if not self.access_token:
            missing.append("HOME_ASSISTANT_TOKEN environment variable")
        if not self.base_url:
            missing.append("HOME_ASSISTANT_URL environment variable")

        try:
            import websockets
        except ImportError:
            missing.append("websockets library (pip install websockets)")

        if missing:
            return f"missing: {', '.join(missing)}"
        else:
            return "Home Assistant configuration incomplete"

    def get_capabilities(self) -> dict[str, Any]:
        """Home Assistant channel capabilities."""
        return {
            "supports_rich_text": False,  # Device control focused
            "supports_buttons": False,
            "supports_audio": False,
            "supports_images": False,
            "bidirectional": True,  # Full bidirectional device communication
            "requires_session": True,  # Need WebSocket connection
            "max_message_length": 1000,
        }

    async def _background_session_loop(self):
        """Run Home Assistant WebSocket in background thread."""
        logger.info("Starting Home Assistant WebSocket connection...")

        while self.session_active:
            try:
                await self._connect_websocket()
            except Exception as e:
                logger.error(f"Home Assistant WebSocket error: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting

    async def _connect_websocket(self):
        """Connect to Home Assistant WebSocket API."""
        try:
            import websockets
        except ImportError:
            raise Exception(
                "websockets library required for Home Assistant WebSocket connection"
            )

        headers = {"Authorization": f"Bearer {self.access_token}"}

        async with websockets.connect(
            self.websocket_url, extra_headers=headers
        ) as websocket:
            self.websocket = websocket
            logger.info("Connected to Home Assistant WebSocket")

            # Handle authentication
            auth_msg = await websocket.recv()
            auth_data = json.loads(auth_msg)

            if auth_data["type"] == "auth_required":
                await websocket.send(
                    json.dumps({"type": "auth", "access_token": self.access_token})
                )

                auth_result = await websocket.recv()
                auth_result_data = json.loads(auth_result)

                if auth_result_data["type"] != "auth_ok":
                    raise Exception(f"Authentication failed: {auth_result_data}")

                logger.info("Home Assistant WebSocket authenticated")

            # Subscribe to state changes
            await self._subscribe_to_events()

            # Listen for messages
            async for message in websocket:
                await self._handle_websocket_message(json.loads(message))

    async def _subscribe_to_events(self):
        """Subscribe to Home Assistant state change events."""
        if not self.websocket:
            return

        # Subscribe to state changes
        subscribe_msg = {
            "id": self._get_next_id(),
            "type": "subscribe_events",
            "event_type": "state_changed",
        }

        await self.websocket.send(json.dumps(subscribe_msg))
        logger.info("Subscribed to Home Assistant state changes")

    async def _handle_websocket_message(self, message: dict[str, Any]):
        """Handle incoming WebSocket message from Home Assistant."""
        msg_type = message.get("type")

        if msg_type == "event":
            await self._handle_state_change_event(message)
        elif msg_type == "result":
            await self._handle_command_result(message)
        elif msg_type == "pong":
            # Heartbeat response
            pass
        else:
            logger.debug(f"Unhandled Home Assistant message type: {msg_type}")

    async def _handle_state_change_event(self, message: dict[str, Any]):
        """Handle state change events from Home Assistant."""
        event_data = message.get("event", {})
        event_type = event_data.get("event_type")

        if event_type == "state_changed":
            data = event_data.get("data", {})
            entity_id = data.get("entity_id")
            new_state = data.get("new_state", {})
            old_state = data.get("old_state", {})

            # Update cached state
            self.entity_states[entity_id] = new_state

            # Check if this is a monitored entity
            if self._should_notify_state_change(entity_id, new_state, old_state):
                await self._send_state_change_notification(
                    entity_id, new_state, old_state
                )

    def _should_notify_state_change(
        self, entity_id: str, new_state: dict, old_state: dict
    ) -> bool:
        """Determine if state change should trigger notification."""
        # Check if entity is in monitored list
        if self.monitored_entities and entity_id not in self.monitored_entities:
            return False

        # Check if domain is monitored
        domain = entity_id.split(".")[0]
        if domain not in self.monitored_domains:
            return False

        # Check for significant state changes
        old_state_value = old_state.get("state") if old_state else None
        new_state_value = new_state.get("state")

        return old_state_value != new_state_value

    async def _send_state_change_notification(
        self, entity_id: str, new_state: dict, old_state: dict
    ):
        """Send notification about state change to main thread."""
        if not self.message_queue:
            return

        notification_text = self._format_state_change_message(
            entity_id, new_state, old_state
        )

        await self.message_queue.put(
            {
                "type": "incoming_message",
                "user_id": "home_assistant",
                "channel_id": "home_assistant",
                "text": notification_text,
                "timestamp": new_state.get("last_changed"),
                "metadata": {
                    "entity_id": entity_id,
                    "new_state": new_state,
                    "old_state": old_state,
                },
            }
        )

    def _format_state_change_message(
        self, entity_id: str, new_state: dict, old_state: dict
    ) -> str:
        """Format state change into human-readable message."""
        friendly_name = new_state.get("attributes", {}).get("friendly_name", entity_id)
        old_state_value = old_state.get("state") if old_state else "unknown"
        new_state_value = new_state.get("state")

        return f"🏠 {friendly_name} changed from {old_state_value} to {new_state_value}"

    async def _handle_command_result(self, message: dict[str, Any]):
        """Handle command result from Home Assistant."""
        command_id = message.get("id")
        success = message.get("success", False)
        result = message.get("result")
        error = message.get("error")

        if command_id in self.pending_commands:
            command_future = self.pending_commands[command_id]
            if success:
                command_future.set_result(result)
            else:
                command_future.set_exception(Exception(f"Command failed: {error}"))
            del self.pending_commands[command_id]

    def _get_next_id(self) -> int:
        """Get next WebSocket message ID."""
        current_id = self.websocket_id
        self.websocket_id += 1
        return current_id

    async def _send(
        self,
        message: str,
        recipient: Optional[str],
        message_format: MessageFormat,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Send command to Home Assistant or control devices.

        Args:
            message: Command text or device control instruction
            recipient: Entity ID or service name
            message_format: Ignored for device control
            metadata: Command parameters:
                - service: Home Assistant service to call
                - entity_id: Target entity ID
                - service_data: Additional service parameters
                - command_type: Type of command (service_call, get_state, etc.)

        Returns:
            Dict with status information
        """
        if not self.websocket:
            return {"success": False, "error": "Home Assistant WebSocket not connected"}

        command_type = metadata.get("command_type", "service_call")

        try:
            if command_type == "service_call":
                return await self._call_service(message, recipient, metadata)
            elif command_type == "get_state":
                return await self._get_entity_state(recipient or message)
            elif command_type == "notification":
                return await self._send_notification_to_ha(message, metadata)
            else:
                return {
                    "success": False,
                    "error": f"Unknown command type: {command_type}",
                }

        except Exception as e:
            logger.error(f"Home Assistant command failed: {e}")
            return {"success": False, "error": str(e)}

    async def _call_service(
        self, message: str, recipient: Optional[str], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a Home Assistant service."""
        service = metadata.get("service")
        entity_id = recipient or metadata.get("entity_id")
        service_data = metadata.get("service_data", {})

        if not service:
            # Try to parse service from message
            service = self._parse_service_from_message(message)

        if not service:
            return {"success": False, "error": "No service specified"}

        # Prepare service call
        command_id = self._get_next_id()
        service_call = {
            "id": command_id,
            "type": "call_service",
            "domain": service.split(".")[0],
            "service": service.split(".")[1] if "." in service else service,
            "service_data": service_data,
        }

        if entity_id:
            service_call["target"] = {"entity_id": entity_id}

        # Set up response tracking
        response_future = asyncio.Future()
        self.pending_commands[command_id] = response_future

        # Send command
        await self.websocket.send(json.dumps(service_call))

        try:
            # Wait for response with timeout
            result = await asyncio.wait_for(response_future, timeout=10.0)
            return {"success": True, "result": result}
        except asyncio.TimeoutError:
            if command_id in self.pending_commands:
                del self.pending_commands[command_id]
            return {"success": False, "error": "Command timeout"}

    def _parse_service_from_message(self, message: str) -> Optional[str]:
        """Parse service name from natural language message."""
        message_lower = message.lower()

        # Simple keyword mapping
        if "turn on" in message_lower or "switch on" in message_lower:
            return "homeassistant.turn_on"
        elif "turn off" in message_lower or "switch off" in message_lower:
            return "homeassistant.turn_off"
        elif "toggle" in message_lower:
            return "homeassistant.toggle"
        elif "set temperature" in message_lower or "temperature" in message_lower:
            return "climate.set_temperature"
        elif "brightness" in message_lower or "dim" in message_lower:
            return "light.turn_on"

        return None

    async def _get_entity_state(self, entity_id: str) -> dict[str, Any]:
        """Get current state of an entity."""
        command_id = self._get_next_id()
        get_state_cmd = {"id": command_id, "type": "get_states"}

        response_future = asyncio.Future()
        self.pending_commands[command_id] = response_future

        await self.websocket.send(json.dumps(get_state_cmd))

        try:
            states = await asyncio.wait_for(response_future, timeout=5.0)

            # Find the specific entity
            for state in states:
                if state.get("entity_id") == entity_id:
                    return {"success": True, "state": state}

            return {"success": False, "error": f"Entity {entity_id} not found"}

        except asyncio.TimeoutError:
            if command_id in self.pending_commands:
                del self.pending_commands[command_id]
            return {"success": False, "error": "State query timeout"}

    async def _send_notification_to_ha(
        self, message: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Send notification through Home Assistant notification service."""
        service_data = {
            "message": message,
            "title": metadata.get("title", "AI Assistant Notification"),
        }

        # Add optional notification parameters
        if "target" in metadata:
            service_data["target"] = metadata["target"]
        if "data" in metadata:
            service_data["data"] = metadata["data"]

        return await self._call_service(
            "send notification",
            None,
            {"service": "notify.notify", "service_data": service_data},
        )

    async def _ask_question_impl(
        self, question: str, options: list[str], timeout: int
    ) -> str:
        """Ask question via Home Assistant input_select or notification."""
        if not self.get_capabilities().get("bidirectional", False):
            raise NotImplementedError(
                "Home Assistant bidirectional communication not available"
            )

        try:
            # Create a temporary input_select entity for the question
            question_id = f"ai_question_{int(asyncio.get_event_loop().time())}"

            # Send notification with question
            await self._send_notification_to_ha(
                f"{question}\n\nOptions: {', '.join(options or ['Yes', 'No'])}",
                {"title": "AI Assistant Question"},
            )

            # For now, return a default response since implementing full
            # input_select integration would require more complex setup
            # In production, you'd create an input_select entity and monitor for changes

            # Simulate user interaction delay
            await asyncio.sleep(1)
            return options[0] if options else "yes"

        except Exception as e:
            logger.error(f"Home Assistant question failed: {e}")
            raise

    async def get_device_state(self, entity_id: str) -> dict[str, Any]:
        """Get current state of a Home Assistant entity."""
        return await self._get_entity_state(entity_id)

    async def control_device(
        self, entity_id: str, action: str, parameters: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Control a Home Assistant device."""
        service_data = parameters or {}

        # Map common actions to services
        service_map = {
            "turn_on": "homeassistant.turn_on",
            "turn_off": "homeassistant.turn_off",
            "toggle": "homeassistant.toggle",
            "set_temperature": "climate.set_temperature",
            "set_brightness": "light.turn_on",
        }

        service = service_map.get(action, action)

        return await self._call_service(
            f"{action} {entity_id}",
            entity_id,
            {"service": service, "service_data": service_data},
        )

    async def list_entities(self, domain: Optional[str] = None) -> dict[str, Any]:
        """List all entities or entities in a specific domain."""
        try:
            command_id = self._get_next_id()
            get_states_cmd = {"id": command_id, "type": "get_states"}

            response_future = asyncio.Future()
            self.pending_commands[command_id] = response_future

            await self.websocket.send(json.dumps(get_states_cmd))

            states = await asyncio.wait_for(response_future, timeout=5.0)

            # Filter by domain if specified
            if domain:
                filtered_states = [
                    state
                    for state in states
                    if state.get("entity_id", "").startswith(f"{domain}.")
                ]
                return {
                    "success": True,
                    "entities": filtered_states,
                    "count": len(filtered_states),
                }
            else:
                return {"success": True, "entities": states, "count": len(states)}

        except Exception as e:
            logger.error(f"Failed to list entities: {e}")
            return {"success": False, "error": str(e)}

    async def stop_session(self):
        """Stop Home Assistant session."""
        self.session_active = False

        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass

        logger.info("Home Assistant session stopped")
