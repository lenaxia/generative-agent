"""Timer Role - Lifecycle-Compatible Pattern

Handles timer management using tools from the central registry.
Integrates with UniversalAgent lifecycle for efficient execution.

Phase 3: Domain-based role with event handlers and intent processors.
"""

import logging
from collections.abc import Callable
from typing import Any, Dict

from common.task_context import TaskContext
from llm_provider.factory import LLMFactory, LLMType

logger = logging.getLogger(__name__)


class TimerRole:
    """Timer role using central tool registry pattern.

    This role declares which tools it needs and provides configuration
    for UniversalAgent lifecycle execution (no separate execute method).
    """

    # Declare required tools (fully qualified names)
    REQUIRED_TOOLS = [
        "timer.set_timer",
        "timer.cancel_timer",
        "timer.list_timers",
    ]

    def __init__(self, tool_registry, llm_factory: LLMFactory):
        """Initialize timer role.

        Args:
            tool_registry: Central ToolRegistry instance
            llm_factory: LLM factory for creating agents (not used in lifecycle pattern)
        """
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.tools = []
        self.name = "timer"

        logger.info(f"TimerRole initialized (tools will load on initialize())")

    async def initialize(self):
        """Load tools from registry.

        This is called after ToolRegistry has been initialized with providers.
        """
        logger.info(f"Loading {len(self.REQUIRED_TOOLS)} tools for TimerRole")

        # Load tools from registry
        self.tools = self.tool_registry.get_tools(self.REQUIRED_TOOLS)

        if len(self.tools) != len(self.REQUIRED_TOOLS):
            missing = len(self.REQUIRED_TOOLS) - len(self.tools)
            logger.warning(f"TimerRole: {missing} tools could not be loaded")

        logger.info(f"TimerRole loaded {len(self.tools)} tools")

    def get_system_prompt(self) -> str:
        """Get system prompt for timer role.

        Returns:
            System prompt string for UniversalAgent
        """
        return """You are a timer management specialist.

Use the available timer tools to set, cancel, and list timers.

Available tools:
- set_timer(duration_seconds, label, deferred_workflow): Set a new timer
- cancel_timer(timer_id): Cancel an active timer
- list_timers(): List all active timers

IMPORTANT: When users request timers with natural language durations:
1. Convert to seconds: "5 minutes" = 300, "1 hour" = 3600, "30 seconds" = 30
2. Call set_timer with the duration_seconds parameter
3. Optionally add a label describing the timer

Examples:
- "Set a timer for 5 minutes" → set_timer(300, "5 minute timer")
- "Remind me in 30 seconds" → set_timer(30, "reminder")
- "Set a 2 hour timer for laundry" → set_timer(7200, "laundry")

Provide clear confirmations when timers are set."""

    def get_llm_type(self) -> LLMType:
        """Get preferred LLM type for this role.

        Returns:
            LLMType for this role (WEAK for simple operations)
        """
        return LLMType.WEAK

    def get_role_config(self) -> dict:
        """Get role configuration for registry.

        Returns:
            Dict with role metadata including fast_reply setting
        """
        return {
            "name": "timer",
            "version": "1.0.0",
            "description": "Timer management with expiry detection",
            "llm_type": "WEAK",
            "fast_reply": True,  # Single-purpose, no multi-step workflows
            "when_to_use": "Set timers, list active timers, cancel timers",
        }

    def get_tools(self):
        """Get loaded tools for this role.

        Returns:
            List of tool instances
        """
        return self.tools

    def get_event_handlers(self) -> dict[str, Callable]:
        """Get event handlers for this role.

        Returns:
            Dict mapping event types to handler functions
        """
        from roles.timer.handlers import handle_heartbeat_monitoring

        return {
            "FAST_HEARTBEAT_TICK": handle_heartbeat_monitoring,
        }

    def get_intent_handlers(self) -> dict[type, Callable]:
        """Get intent handlers for this role.

        Returns:
            Dict mapping Intent classes to processor functions
        """
        from roles.timer.handlers import (
            TimerCancellationIntent,
            TimerCreationIntent,
            TimerExpiryIntent,
            TimerListingIntent,
            process_timer_cancellation_intent,
            process_timer_creation_intent,
            process_timer_expiry_intent,
            process_timer_listing_intent,
        )

        return {
            TimerCreationIntent: process_timer_creation_intent,
            TimerCancellationIntent: process_timer_cancellation_intent,
            TimerListingIntent: process_timer_listing_intent,
            TimerExpiryIntent: process_timer_expiry_intent,
        }
