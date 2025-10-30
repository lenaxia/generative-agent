"""
EventHandlerContext for clean dependency injection in event handlers.

This module provides a context object pattern to avoid crowded handler signatures
and provide clean access to all necessary dependencies for event processing.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from common.event_handler_llm import EventHandlerLLM
from common.message_bus import MessageBus

logger = logging.getLogger(__name__)


@dataclass
class EventHandlerContext:
    """Context object for event handlers with all necessary dependencies.

    This class follows the Context Object pattern to provide clean dependency
    injection for event handlers, avoiding crowded constructor signatures.

    Usage:
        async def my_handler(event_data: Dict[str, Any], ctx: EventHandlerContext):
            user_id = ctx.get_user_id()
            await ctx.create_workflow("do something")
            ctx.publish_event("MY_EVENT", {"status": "completed"})
    """

    llm: EventHandlerLLM
    workflow_engine: Any  # WorkflowEngine (avoiding circular import)
    communication_manager: Any  # CommunicationManager (avoiding circular import)
    message_bus: MessageBus
    execution_context: dict[str, Any]

    def get_user_id(self) -> str:
        """Get user ID from execution context."""
        return self.llm.get_user_id()

    def get_channel(self) -> str:
        """Get channel from execution context."""
        return self.llm.get_channel()

    def get_original_request(self) -> str:
        """Get original user request."""
        return self.llm.get_original_request()

    def get_context(self, key: str = None) -> Any:
        """Get execution context data."""
        return self.llm.get_context(key)

    async def create_workflow(
        self, instruction: str, context: dict[str, Any] | None = None
    ) -> str:
        """Create a new workflow with the given instruction.

        Args:
            instruction: Workflow instruction
            context: Optional context override (uses execution_context by default)

        Returns:
            Workflow ID
        """
        workflow_context = context or self.execution_context
        return await self.workflow_engine.start_workflow(instruction=instruction)

    async def send_notification(
        self,
        message: str,
        channels: list[str] | None = None,
        recipient: str | None = None,
    ):
        """Send notification through communication manager.

        Args:
            message: Notification message
            channels: List of channel types (defaults to SLACK)
            recipient: Notification recipient (defaults to execution context channel)
        """

        # Default values from execution context
        default_channel = channels[0] if channels else "slack"
        default_recipient = recipient or self.get_channel()

        # Use current route_message API instead of legacy send_notification
        context = {
            "channel_id": default_channel,
            "recipient": default_recipient,
            "message_format": "plain_text",
            "message_type": "notification",
            "metadata": self.execution_context,
        }

        await self.communication_manager.route_message(message, context)

    def publish_event(self, event_type: str, event_data: dict[str, Any]):
        """Publish an event to the MessageBus.

        Args:
            event_type: Type of event to publish
            event_data: Event data payload
        """
        self.message_bus.publish(self, event_type, event_data)
        logger.info(
            f"Published {event_type} event: {event_data.get('timer_id', 'unknown')}"
        )

    async def invoke_llm(self, prompt: str, model_type: str = "WEAK") -> str:
        """Invoke LLM with the given prompt.

        Args:
            prompt: Prompt to send to LLM
            model_type: Model strength to use

        Returns:
            LLM response
        """
        return await self.llm.invoke(prompt, model_type)

    async def parse_json(self, prompt: str, model_type: str = "WEAK") -> dict[str, Any]:
        """Parse JSON response from LLM.

        Args:
            prompt: Prompt that should return JSON
            model_type: Model strength to use

        Returns:
            Parsed JSON dict
        """
        return await self.llm.parse_json(prompt, model_type)

    async def quick_decision(self, question: str, options: list[str] = None) -> str:
        """Make a quick decision using LLM.

        Args:
            question: Question to ask
            options: Optional list of valid options

        Returns:
            LLM decision
        """
        return await self.llm.quick_decision(question, options)
