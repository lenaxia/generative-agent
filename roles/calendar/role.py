"""Calendar Role - Lifecycle-Compatible Pattern

Handles calendar operations using tools from the central registry.
Integrates with UniversalAgent lifecycle for efficient execution.
"""

import logging
from typing import Any

from common.task_context import TaskContext
from llm_provider.factory import LLMFactory, LLMType

logger = logging.getLogger(__name__)


class CalendarRole:
    """Calendar role using central tool registry pattern.

    This role declares which tools it needs and provides configuration
    for UniversalAgent lifecycle execution (no separate execute method).
    """

    # Declare required tools (fully qualified names)
    REQUIRED_TOOLS = [
        "calendar.get_schedule",
        "calendar.add_calendar_event",
    ]

    def __init__(self, tool_registry, llm_factory: LLMFactory):
        """Initialize calendar role.

        Args:
            tool_registry: Central ToolRegistry instance
            llm_factory: LLM factory for creating agents (not used in lifecycle pattern)
        """
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.tools = []
        self.name = "calendar"

        logger.info(f"CalendarRole initialized (tools will load on initialize())")

    async def initialize(self):
        """Load tools from registry.

        This is called after ToolRegistry has been initialized with providers.
        """
        logger.info(f"Loading {len(self.REQUIRED_TOOLS)} tools for CalendarRole")

        # Load tools from registry
        self.tools = self.tool_registry.get_tools(self.REQUIRED_TOOLS)

        if len(self.tools) != len(self.REQUIRED_TOOLS):
            missing = len(self.REQUIRED_TOOLS) - len(self.tools)
            logger.warning(f"CalendarRole: {missing} tools could not be loaded")

        logger.info(f"CalendarRole loaded {len(self.tools)} tools")

    def get_system_prompt(self) -> str:
        """Get system prompt for calendar role.

        Returns:
            System prompt string for UniversalAgent
        """
        return """You are a calendar management specialist.

Use the available calendar tools to check schedules and manage events.

Available tools:
- get_schedule(start_date, end_date, days): Get calendar events
- add_calendar_event(summary, start, end, description, location): Create new event

Provide clear confirmations when creating events and helpful summaries when viewing schedules."""

    def get_llm_type(self) -> LLMType:
        """Get preferred LLM type for this role.

        Returns:
            LLMType for this role (DEFAULT for calendar operations)
        """
        return LLMType.DEFAULT

    def get_role_config(self) -> dict:
        """Get role configuration for registry.

        Returns:
            Dict with role metadata including fast_reply setting
        """
        return {
            "name": "calendar",
            "version": "1.0.0",
            "description": "Calendar and scheduling management",
            "llm_type": "DEFAULT",
            "fast_reply": True,  # Single-purpose information retrieval
            "when_to_use": "Schedule management, calendar queries, event planning",
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
        from roles.calendar.handlers import handle_calendar_request

        return {
            "CALENDAR_REQUEST": handle_calendar_request,
        }

    def get_intent_handlers(self):
        """Get intent handlers for this role.

        Returns:
            Dict mapping intent classes to processor functions
        """
        from roles.calendar.handlers import (
            CalendarIntent,
            process_calendar_intent,
        )

        return {
            CalendarIntent: process_calendar_intent,
        }
