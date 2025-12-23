"""Smart Home Role - Lifecycle-Compatible Pattern

Handles smart home control using tools from the central registry.
Integrates with UniversalAgent lifecycle for efficient execution.
"""

import logging
from typing import Any

from common.task_context import TaskContext
from llm_provider.factory import LLMFactory, LLMType

logger = logging.getLogger(__name__)


class SmartHomeRole:
    """Smart home role using central tool registry pattern.

    This role declares which tools it needs and provides configuration
    for UniversalAgent lifecycle execution (no separate execute method).
    """

    # Declare required tools (fully qualified names)
    REQUIRED_TOOLS = [
        "smart_home.ha_call_service",
        "smart_home.ha_get_state",
        "smart_home.ha_list_entities",
    ]

    def __init__(self, tool_registry, llm_factory: LLMFactory):
        """Initialize smart home role.

        Args:
            tool_registry: Central ToolRegistry instance
            llm_factory: LLM factory for creating agents (not used in lifecycle pattern)
        """
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.tools = []
        self.name = "smart_home"

        logger.info(f"SmartHomeRole initialized (tools will load on initialize())")

    async def initialize(self):
        """Load tools from registry.

        This is called after ToolRegistry has been initialized with providers.
        """
        logger.info(f"Loading {len(self.REQUIRED_TOOLS)} tools for SmartHomeRole")

        # Load tools from registry
        self.tools = self.tool_registry.get_tools(self.REQUIRED_TOOLS)

        if len(self.tools) != len(self.REQUIRED_TOOLS):
            missing = len(self.REQUIRED_TOOLS) - len(self.tools)
            logger.warning(f"SmartHomeRole: {missing} tools could not be loaded")

        logger.info(f"SmartHomeRole loaded {len(self.tools)} tools")

    def get_system_prompt(self) -> str:
        """Get system prompt for smart home role.

        Returns:
            System prompt string for UniversalAgent
        """
        return """You are a smart home control specialist.

Use the available Home Assistant tools to control and monitor smart home devices.

Available tools:
- ha_call_service(domain, service, entity_id, **service_data): Call a service
- ha_get_state(entity_id): Get entity state
- ha_list_entities(domain): List available entities

Common operations:
- Turn on light: ha_call_service("light", "turn_on", "light.living_room")
- Set brightness: ha_call_service("light", "turn_on", "light.bedroom", brightness=255)
- Check temperature: ha_get_state("sensor.temperature")
- Set thermostat: ha_call_service("climate", "set_temperature", "climate.bedroom", temperature=72)

Provide clear confirmations of smart home actions."""

    def get_llm_type(self) -> LLMType:
        """Get preferred LLM type for this role.

        Returns:
            LLMType for this role (DEFAULT for home control)
        """
        return LLMType.DEFAULT

    def get_role_config(self) -> dict:
        """Get role configuration for registry.

        Returns:
            Dict with role metadata including fast_reply setting
        """
        return {
            "name": "smart_home",
            "version": "1.0.0",
            "description": "Smart home control with Home Assistant integration",
            "llm_type": "DEFAULT",
            "fast_reply": True,  # Most operations are simple single commands
            "when_to_use": "Control lights, switches, climate, sensors via Home Assistant",
        }

    def get_tools(self):
        """Get loaded tools for this role.

        Returns:
            List of tool instances
        """
        return self.tools

    def get_event_handlers(self):
        """Get event handlers for this role.

        Returns:
            Dict mapping event types to handler functions
        """
        from roles.smart_home.handlers import (
            handle_smart_home_request,
            handle_device_discovery,
        )

        return {
            "SMART_HOME_REQUEST": handle_smart_home_request,
            "DEVICE_DISCOVERY": handle_device_discovery,
        }

    def get_intent_handlers(self):
        """Get intent handlers for this role.

        Returns:
            Dict mapping intent classes to processor functions
        """
        from roles.smart_home.handlers import (
            HomeAssistantServiceIntent,
            HomeAssistantStateIntent,
            SmartHomeControlIntent,
            process_home_assistant_service_intent,
            process_home_assistant_state_intent,
            process_smart_home_control_intent,
        )

        return {
            HomeAssistantServiceIntent: process_home_assistant_service_intent,
            HomeAssistantStateIntent: process_home_assistant_state_intent,
            SmartHomeControlIntent: process_smart_home_control_intent,
        }
