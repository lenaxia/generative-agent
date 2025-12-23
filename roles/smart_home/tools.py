"""Smart Home Domain Tools

Provides Home Assistant integration tools for the dynamic agent system.
Tools interact with Home Assistant via MCP server or REST API.

Extracted from: roles/core_smart_home.py
"""

import logging
from typing import Any

from strands import tool

logger = logging.getLogger(__name__)


def create_smart_home_tools(home_assistant_provider: Any) -> list:
    """Create smart home domain tools.

    Args:
        home_assistant_provider: Home Assistant provider instance (MCP or REST API)

    Returns:
        List of tool functions for smart home domain
    """
    tools = [
        ha_call_service,
        ha_get_state,
        ha_list_entities,
    ]

    logger.info(f"Created {len(tools)} smart home tools")
    return tools


# ACTION TOOLS


@tool
def ha_call_service(
    domain: str, service: str, entity_id: str | None = None, **service_data
) -> dict[str, Any]:
    """Call a Home Assistant service.

    Action tool - controls smart home devices (has side effects).

    Args:
        domain: Home Assistant domain (e.g., "light", "switch", "climate")
        service: Service to call (e.g., "turn_on", "turn_off", "set_temperature")
        entity_id: Optional entity ID to target
        **service_data: Additional service-specific data

    Examples:
        ha_call_service("light", "turn_on", "light.living_room", brightness=255)
        ha_call_service("climate", "set_temperature", "climate.bedroom", temperature=72)

    Returns:
        Dict with success status and message
    """
    logger.info(f"Calling HA service: {domain}.{service} for {entity_id}")

    # Return intent for Home Assistant service call
    # The infrastructure will process this via MCP or REST API
    return {
        "success": True,
        "message": f"Called {domain}.{service}" + (f" on {entity_id}" if entity_id else ""),
        "intent": {
            "type": "HomeAssistantServiceIntent",
            "domain": domain,
            "service": service,
            "entity_id": entity_id,
            "service_data": service_data,
        },
    }


# QUERY TOOLS


@tool
def ha_get_state(entity_id: str) -> dict[str, Any]:
    """Get the state of a Home Assistant entity.

    Query tool - reads entity state (no side effects).

    Args:
        entity_id: Entity ID to query (e.g., "light.living_room", "sensor.temperature")

    Returns:
        Dict with success status and entity state data
    """
    logger.info(f"Getting HA state for: {entity_id}")

    # Return intent for state query
    # The infrastructure will fetch from MCP or REST API
    return {
        "success": True,
        "message": f"Querying state for {entity_id}",
        "intent": {
            "type": "HomeAssistantStateQueryIntent",
            "entity_id": entity_id,
        },
    }


@tool
def ha_list_entities(domain: str | None = None) -> dict[str, Any]:
    """List Home Assistant entities, optionally filtered by domain.

    Query tool - lists available entities (no side effects).

    Args:
        domain: Optional domain to filter by (e.g., "light", "switch", "sensor")

    Returns:
        Dict with success status and entity list
    """
    logger.info(f"Listing HA entities" + (f" for domain: {domain}" if domain else ""))

    # Return intent for entity listing
    return {
        "success": True,
        "message": "Listing entities" + (f" in domain {domain}" if domain else ""),
        "intent": {
            "type": "HomeAssistantEntityListIntent",
            "domain": domain,
        },
    }
