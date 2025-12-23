"""Smart Home Role Handlers - Phase 3 Domain Pattern

Event handlers, intent processors, and helper functions for smart_home role.
Includes Home Assistant MCP integration support.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from common.event_context import LLMSafeEventContext
from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)


# INTENT DEFINITIONS
@dataclass
class HomeAssistantServiceIntent(Intent):
    """Home Assistant service call intent - owned by smart_home role."""

    domain: str  # "light", "switch", "climate", etc.
    service: str  # "turn_on", "turn_off", "set_brightness", etc.
    entity_id: str | None = None
    service_data: dict[str, Any] | None = None
    user_id: str | None = None
    channel_id: str | None = None
    event_context: dict[str, Any] | None = None

    def validate(self) -> bool:
        """Validate Home Assistant service intent parameters."""
        return (
            bool(self.domain and self.service)
            and len(self.domain.strip()) > 0
            and len(self.service.strip()) > 0
        )


@dataclass
class HomeAssistantStateIntent(Intent):
    """Home Assistant state query intent - owned by smart_home role."""

    entity_id: str | None = None
    domain: str | None = None
    operation: str = "get_state"  # "get_state", "list_entities"
    user_id: str | None = None
    channel_id: str | None = None

    def validate(self) -> bool:
        """Validate Home Assistant state intent parameters."""
        return bool(self.operation in ["get_state", "list_entities"])


@dataclass
class SmartHomeControlIntent(Intent):
    """Smart home control coordination intent - owned by smart_home role."""

    action: str
    target_entity: str | None = None
    parameters: dict[str, Any] | None = None
    user_id: str | None = None
    channel_id: str | None = None
    event_context: dict[str, Any] | None = None

    def validate(self) -> bool:
        """Validate smart home control intent parameters."""
        valid_actions = [
            "control_device",
            "query_state",
            "discover_entities",
            "execute_scene",
        ]
        return bool(self.action and self.action in valid_actions)


# EVENT HANDLERS
def handle_smart_home_request(
    event_data: Any, context: LLMSafeEventContext
) -> list[Intent]:
    """LLM-SAFE: Pure function for smart home request events."""
    try:
        # Parse event data
        action, entity_id, parameters = _parse_smart_home_event_data(event_data)

        # Check if parsing failed
        if action == "parse_error":
            logger.error(f"Smart home handler error: {entity_id}")
            return [
                NotificationIntent(
                    message=f"Smart home processing error: {entity_id}",
                    channel=context.get_safe_channel(),
                    priority="high",
                    notification_type="error",
                )
            ]

        # Create intents for successful parsing
        intents = [
            SmartHomeControlIntent(
                action="control_device",
                target_entity=entity_id,
                parameters=parameters or {},
                user_id=context.user_id,
                channel_id=context.channel_id,
                event_context=context.to_dict(),
            ),
            AuditIntent(
                action="smart_home_request",
                details={
                    "action": action,
                    "entity_id": entity_id,
                    "parameters": parameters,
                    "processed_at": time.time(),
                },
                user_id=context.user_id,
                severity="info",
            ),
        ]

        return intents

    except Exception as e:
        logger.error(f"Smart home handler error: {e}")
        return [
            NotificationIntent(
                message=f"Smart home processing error: {e}",
                channel=context.get_safe_channel(),
                priority="high",
                notification_type="error",
            )
        ]


def handle_device_discovery(
    event_data: Any, context: LLMSafeEventContext
) -> list[Intent]:
    """LLM-SAFE: Pure function for device discovery events."""
    try:
        domain = _parse_device_discovery_event(event_data)

        return [
            HomeAssistantStateIntent(
                domain=domain,
                operation="list_entities",
                user_id=context.user_id,
                channel_id=context.channel_id,
            ),
            AuditIntent(
                action="device_discovery",
                details={
                    "domain": domain,
                    "processed_at": time.time(),
                },
                user_id=context.user_id,
                severity="info",
            ),
        ]

    except Exception as e:
        logger.error(f"Device discovery error: {e}")
        return [
            NotificationIntent(
                message=f"Device discovery error: {e}",
                channel=context.get_safe_channel(),
                priority="medium",
                notification_type="warning",
            )
        ]


# HELPER FUNCTIONS
def _parse_smart_home_event_data(event_data: Any) -> tuple[str, str, dict[str, Any]]:
    """LLM-SAFE: Parse smart home event data with error handling."""
    try:
        if isinstance(event_data, dict):
            return (
                event_data.get("action", "control_device"),
                event_data.get("entity_id", ""),
                event_data.get("parameters", {}),
            )
        elif isinstance(event_data, list) and len(event_data) >= 2:
            return (
                str(event_data[0]),
                str(event_data[1]),
                (
                    event_data[2]
                    if len(event_data) > 2 and isinstance(event_data[2], dict)
                    else {}
                ),
            )
        else:
            return "control_device", "", {}
    except Exception as e:
        return "parse_error", f"Parse error: {e}", {}


def _parse_device_discovery_event(event_data: Any) -> str:
    """LLM-SAFE: Parse device discovery event data."""
    try:
        if isinstance(event_data, dict):
            return event_data.get("domain", "")
        elif isinstance(event_data, str):
            return event_data
        else:
            return ""
    except Exception as e:
        return f"parse_error: {e}"


def _get_safe_channel(context) -> str:
    """LLM-SAFE: Get safe channel from context."""
    if hasattr(context, "get_safe_channel") and callable(context.get_safe_channel):
        try:
            result = context.get_safe_channel()
            return str(result) if result else "console"
        except TypeError:
            pass
    if hasattr(context, "channel_id") and context.channel_id:
        return str(context.channel_id)
    else:
        return "console"


# INTENT PROCESSORS
async def process_home_assistant_service_intent(intent: HomeAssistantServiceIntent):
    """Process Home Assistant service call intents - handles actual MCP operations."""
    logger.info(f"Processing HA service call: {intent.domain}.{intent.service}")

    try:
        # This would integrate with MCP client to make actual Home Assistant calls
        # For now, log the intent processing
        logger.info(
            f"Would call HA service: {intent.domain}.{intent.service} on {intent.entity_id}"
        )
        logger.info(f"Service data: {intent.service_data}")
        logger.info(f"Context: {intent.event_context}")

        # TODO: In full implementation, this would:
        # 1. Get MCP client for home_assistant server
        # 2. Call the appropriate MCP tool (call_service)
        # 3. Handle response and errors
        # 4. Update device state tracking
        # 5. Send confirmation notifications

    except Exception as e:
        logger.error(f"Home Assistant service call failed: {e}")


async def process_home_assistant_state_intent(intent: HomeAssistantStateIntent):
    """Process Home Assistant state query intents - handles actual MCP operations."""
    logger.info(f"Processing HA state query: {intent.operation}")

    try:
        if intent.operation == "get_state" and intent.entity_id:
            logger.info(f"Would get state for: {intent.entity_id}")
        elif intent.operation == "list_entities":
            logger.info(f"Would list entities in domain: {intent.domain}")

        # TODO: In full implementation, this would:
        # 1. Get MCP client for home_assistant server
        # 2. Call appropriate MCP tool (get_state or list_entities)
        # 3. Process and format response
        # 4. Cache entity information
        # 5. Return structured data

    except Exception as e:
        logger.error(f"Home Assistant state query failed: {e}")


async def process_smart_home_control_intent(intent: SmartHomeControlIntent):
    """Process smart home control coordination intents."""
    logger.info(f"Processing smart home control: {intent.action}")

    try:
        # Coordinate between different Home Assistant operations
        # Handle complex multi-step operations
        # Manage device state consistency
        logger.info(f"Control action: {intent.action} on {intent.target_entity}")
        logger.info(f"Parameters: {intent.parameters}")

        # TODO: Implement smart home control coordination logic

    except Exception as e:
        logger.error(f"Smart home control coordination failed: {e}")


# PRE/POST PROCESSORS (optional, for MCP integration)
async def fetch_home_assistant_entities(parameters: dict[str, Any]) -> dict[str, Any]:
    """Pre-processor: Fetch Home Assistant entities before LLM call."""
    try:
        domain = parameters.get("domain")
        entity_id = parameters.get("entity_id")

        logger.info(f"Pre-fetching HA entities - domain: {domain}, entity: {entity_id}")

        # TODO: In full implementation, this would:
        # 1. Connect to Home Assistant MCP server
        # 2. Fetch available entities and their current states
        # 3. Filter by domain if specified
        # 4. Return structured data for prompt injection

        return {
            "success": True,
            "entities": [],  # Would contain actual entity data
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to pre-fetch HA entities: {e}")
        return {"success": False, "error": str(e), "entities": []}


async def format_home_assistant_response(
    llm_result: str, context, pre_data: dict
) -> str:
    """Post-processor: Format LLM result with Home Assistant context."""
    try:
        # TODO: Add device state information to response
        # Format entity names in user-friendly way
        # Include current status and confirmation

        return llm_result  # Would enhance with HA-specific formatting

    except Exception as e:
        logger.error(f"Failed to format HA response: {e}")
        return llm_result
