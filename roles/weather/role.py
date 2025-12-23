"""Weather Role - Lifecycle-Compatible Pattern

Handles weather queries using tools from the central registry.
Integrates with UniversalAgent lifecycle for efficient execution.
"""

import logging
from typing import Any

from common.task_context import TaskContext
from llm_provider.factory import LLMFactory, LLMType

logger = logging.getLogger(__name__)


class WeatherRole:
    """Weather role using central tool registry pattern.

    This role declares which tools it needs and provides configuration
    for UniversalAgent lifecycle execution (no separate execute method).
    """

    # Declare required tools (fully qualified names)
    REQUIRED_TOOLS = [
        "weather.get_current_weather",
        "weather.get_forecast",
    ]

    def __init__(self, tool_registry, llm_factory: LLMFactory):
        """Initialize weather role.

        Args:
            tool_registry: Central ToolRegistry instance
            llm_factory: LLM factory for creating agents (not used in lifecycle pattern)
        """
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.tools = []
        self.name = "weather"

        logger.info(f"WeatherRole initialized (tools will load on initialize())")

    async def initialize(self):
        """Load tools from registry.

        This is called after ToolRegistry has been initialized with providers.
        """
        logger.info(f"Loading {len(self.REQUIRED_TOOLS)} tools for WeatherRole")

        # Load tools from registry
        self.tools = self.tool_registry.get_tools(self.REQUIRED_TOOLS)

        if len(self.tools) != len(self.REQUIRED_TOOLS):
            missing = len(self.REQUIRED_TOOLS) - len(self.tools)
            logger.warning(f"WeatherRole: {missing} tools could not be loaded")

        logger.info(f"WeatherRole loaded {len(self.tools)} tools")

    def get_system_prompt(self) -> str:
        """Get system prompt for weather role.

        Returns:
            System prompt string for UniversalAgent
        """
        return """You are a weather specialist providing accurate weather information.

Use the available weather tools to get current conditions or forecasts.
Provide clear, helpful weather information based on the data returned by the tools.

Available tools:
- get_current_weather(location): Get current weather for a location
- get_forecast(location, days): Get extended forecast (default 7 days)

Be concise and informative in your responses."""

    def get_llm_type(self) -> LLMType:
        """Get preferred LLM type for this role.

        Returns:
            LLMType for this role (WEAK for simple queries)
        """
        return LLMType.WEAK

    def get_role_config(self) -> dict:
        """Get role configuration for registry.

        Returns:
            Dict with role metadata including fast_reply setting
        """
        return {
            "name": "weather",
            "version": "1.0.0",
            "description": "Weather information and forecasts",
            "llm_type": "WEAK",
            "fast_reply": True,  # Single-purpose information retrieval
            "when_to_use": "Get current weather conditions, forecasts, location-based weather",
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
        from roles.weather.handlers import (
            handle_weather_request,
            handle_weather_data_processing,
        )

        return {
            "WEATHER_REQUEST": handle_weather_request,
            "WEATHER_DATA_PROCESSING": handle_weather_data_processing,
        }

    def get_intent_handlers(self):
        """Get intent handlers for this role.

        Returns:
            Dict mapping intent classes to processor functions
        """
        from roles.weather.handlers import (
            WeatherIntent,
            WeatherDataIntent,
            process_weather_intent,
            process_weather_data_intent,
        )

        return {
            WeatherIntent: process_weather_intent,
            WeatherDataIntent: process_weather_data_intent,
        }
