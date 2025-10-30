"""
Reusable Event Subscriber Pattern for Role-Based Event Handling.

This module provides a standardized way for roles to subscribe to and handle
events from the MessageBus, enabling delayed actions, cross-role communication,
and complex event-driven workflows.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from collections.abc import Callable

from common.message_bus import MessageBus, MessageType

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions that can be handled by event subscribers."""

    NOTIFICATION = "notification"  # Simple notification
    WORKFLOW = "workflow"  # Create new workflow
    DIRECT_ACTION = "direct_action"  # Execute specific role action
    CONDITIONAL_WORKFLOW = "conditional"  # Workflow with conditions


class EventSubscriber:
    """Reusable pattern for roles to subscribe to and handle events.

    This class provides a standardized way for roles to:
    1. Subscribe to MessageBus events
    2. Parse event data with LLM assistance
    3. Create workflows or execute actions based on events
    4. Handle delivery guarantees and error recovery

    Usage:
        # In role lifecycle or initialization:
        subscriber = EventSubscriber("timer", workflow_engine, universal_agent)
        await subscriber.subscribe_to_events(message_bus)
    """

    def __init__(
        self,
        role_name: str,
        workflow_engine,
        universal_agent,
        message_bus: MessageBus | None = None,
    ):
        """Initialize event subscriber for a specific role.

        Args:
            role_name: Name of the role (e.g., "timer", "calendar", "smart_home")
            workflow_engine: WorkflowEngine instance for creating workflows
            universal_agent: UniversalAgent instance for LLM calls
            message_bus: Optional MessageBus instance
        """
        self.role_name = role_name
        self.workflow_engine = workflow_engine
        self.universal_agent = universal_agent
        self.message_bus = message_bus
        self.event_handlers = {}

        # Default event handlers for common patterns
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default event handlers for common patterns."""
        # Timer expiry handler
        self.event_handlers[MessageType.TIMER_EXPIRED] = self.handle_delayed_action

        # Could add more default handlers:
        # self.event_handlers[MessageType.CALENDAR_EVENT] = self.handle_calendar_action
        # self.event_handlers[MessageType.SENSOR_TRIGGERED] = self.handle_sensor_action

    async def subscribe_to_events(self, message_bus: MessageBus):
        """Subscribe to relevant events on the message bus.

        Args:
            message_bus: MessageBus instance to subscribe to
        """
        self.message_bus = message_bus

        for message_type, handler in self.event_handlers.items():
            self.message_bus.subscribe(self, message_type, handler)
            logger.info(f"Role '{self.role_name}' subscribed to {message_type.value}")

    async def handle_delayed_action(self, event_data: dict[str, Any]):
        """Handle delayed actions from events (e.g., timer expiry, calendar events).

        This is the core method that:
        1. Parses the original request with current context
        2. Determines if it's a notification or workflow
        3. Executes the appropriate action

        Args:
            event_data: Event data containing original request and context
        """
        try:
            logger.info(
                f"Handling delayed action for role '{self.role_name}': {event_data.get('original_request', 'Unknown')}"
            )

            # Parse the action with LLM using current context
            parsed_action = await self._parse_delayed_action(event_data)

            # Execute based on action type
            if parsed_action["action_type"] == ActionType.NOTIFICATION.value:
                await self._handle_notification_action(parsed_action, event_data)
            elif parsed_action["action_type"] == ActionType.WORKFLOW.value:
                await self._handle_workflow_action(parsed_action, event_data)
            elif parsed_action["action_type"] == ActionType.DIRECT_ACTION.value:
                await self._handle_direct_action(parsed_action, event_data)
            else:
                logger.warning(f"Unknown action type: {parsed_action['action_type']}")
                # Fallback to notification
                await self._handle_notification_fallback(event_data)

        except Exception as e:
            logger.error(f"Error handling delayed action: {e}")
            # Always fallback to basic notification on error
            await self._handle_notification_fallback(event_data)

    async def _parse_delayed_action(self, event_data: dict[str, Any]) -> dict[str, Any]:
        """Parse delayed action using LLM with preserved context.

        Args:
            event_data: Original event data with request and context

        Returns:
            Parsed action with type and instructions
        """
        original_request = event_data.get("original_request", "")
        execution_context = event_data.get("execution_context", {})

        # Create parsing prompt with rich context
        parsing_prompt = f"""
Parse this delayed action request and determine how to execute it:

Original Request: "{original_request}"

Execution Context:
- User ID: {execution_context.get('user_id', 'unknown')}
- Channel: {execution_context.get('channel', 'unknown')}
- Room/Device Context: {execution_context.get('device_context', {})}
- Original Timestamp: {execution_context.get('timestamp', 'unknown')}
- Source: {execution_context.get('source', 'unknown')}

Current Time: {datetime.now().isoformat()}

Determine:
1. Action Type: "notification", "workflow", "direct_action", or "conditional"
2. Workflow Instruction: Clear instruction for workflow execution
3. Target: Where/what to execute the action on

Respond in JSON format:
{{
    "action_type": "workflow",
    "workflow_instruction": "turn on bedroom lights to 50% brightness",
    "target_context": {{"room": "bedroom", "devices": ["bedroom_lights"]}},
    "fallback_notification": "Reminder: turn on bedroom lights"
}}
"""

        # Use router role for parsing (lightweight)
        try:
            response = await self.universal_agent.assume_role("router").execute_task(
                parsing_prompt, context=execution_context
            )

            # Parse JSON response
            parsed_action = json.loads(response)
            return parsed_action

        except Exception as e:
            logger.error(f"Error parsing delayed action: {e}")
            # Return safe fallback
            return {
                "action_type": "notification",
                "workflow_instruction": f"Reminder: {original_request}",
                "fallback_notification": f"Timer reminder: {original_request}",
            }

    async def _handle_notification_action(
        self, parsed_action: dict[str, Any], event_data: dict[str, Any]
    ):
        """Handle simple notification actions."""

        # Get communication manager (would need to be injected or accessed)
        # This is a design decision - how to access CommunicationManager
        notification_message = parsed_action.get(
            "workflow_instruction", "Timer notification"
        )
        execution_context = event_data.get("execution_context", {})

        # Send notification to original channel
        # Note: This requires access to CommunicationManager - architectural decision needed
        logger.info(f"Sending notification: {notification_message}")

    async def _handle_workflow_action(
        self, parsed_action: dict[str, Any], event_data: dict[str, Any]
    ):
        """Handle workflow creation actions."""
        workflow_instruction = parsed_action.get("workflow_instruction", "")
        execution_context = event_data.get("execution_context", {})

        if not workflow_instruction:
            logger.error("No workflow instruction provided")
            await self._handle_notification_fallback(event_data)
            return

        # Create new workflow with preserved context
        try:
            workflow_id = await self.workflow_engine.start_workflow(
                instruction=workflow_instruction, context=execution_context
            )
            logger.info(
                f"Created workflow {workflow_id} from delayed action: {workflow_instruction}"
            )

        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            await self._handle_notification_fallback(event_data)

    async def _handle_direct_action(
        self, parsed_action: dict[str, Any], event_data: dict[str, Any]
    ):
        """Handle direct role actions without creating full workflows."""
        target_role = parsed_action.get("target_role", "default")
        action_params = parsed_action.get("action_params", {})
        execution_context = event_data.get("execution_context", {})

        try:
            # Execute direct action on target role
            result = await self.universal_agent.assume_role(target_role).execute_task(
                action_params, context=execution_context
            )
            logger.info(f"Executed direct action on role '{target_role}': {result}")

        except Exception as e:
            logger.error(f"Failed to execute direct action: {e}")
            await self._handle_notification_fallback(event_data)

    async def _handle_notification_fallback(self, event_data: dict[str, Any]):
        """Fallback notification when other actions fail."""
        original_request = event_data.get("original_request", "Timer expired")
        execution_context = event_data.get("execution_context", {})

        # Simple fallback notification
        fallback_message = f"Reminder: {original_request}"
        logger.info(f"Sending fallback notification: {fallback_message}")

        # Note: This would need access to CommunicationManager
        # Could be injected in constructor or accessed via workflow_engine

    def add_custom_handler(self, message_type: MessageType, handler: Callable):
        """Add custom event handler for specific message types.

        Args:
            message_type: MessageType to handle
            handler: Async function to handle the event
        """
        self.event_handlers[message_type] = handler
        logger.info(
            f"Added custom handler for {message_type.value} in role '{self.role_name}'"
        )

    async def unsubscribe_from_events(self):
        """Unsubscribe from all events on the message bus."""
        if self.message_bus:
            for message_type in self.event_handlers.keys():
                # Note: MessageBus would need unsubscribe method
                logger.info(f"Unsubscribed from {message_type.value}")


# Usage example for timer role:
class TimerEventSubscriber(EventSubscriber):
    """Timer-specific event subscriber with enhanced timer handling."""

    def __init__(self, workflow_engine, universal_agent, communication_manager):
        super().__init__("timer", workflow_engine, universal_agent)
        self.communication_manager = communication_manager

    async def handle_timer_expiry_with_context(self, timer_data: dict[str, Any]):
        """Enhanced timer expiry handler with communication manager access."""
        try:
            # Use parent class parsing
            parsed_action = await self._parse_delayed_action(timer_data)

            if parsed_action["action_type"] == ActionType.NOTIFICATION.value:
                # Handle notification directly with communication manager
                await self.communication_manager.send_notification(
                    message=parsed_action["workflow_instruction"],
                    channels=["slack"],  # or determine from context
                    recipient=timer_data["execution_context"].get("channel"),
                    metadata=timer_data["execution_context"],
                )
            else:
                # Handle workflow actions
                await self._handle_workflow_action(parsed_action, timer_data)

        except Exception as e:
            logger.error(f"Timer expiry handling failed: {e}")
            # Send basic notification as fallback
            await self.communication_manager.send_notification(
                message=f"Timer reminder: {timer_data.get('original_request', 'Timer expired')}",
                channels=["slack"],
                recipient=timer_data["execution_context"].get("channel"),
                metadata=timer_data["execution_context"],
            )
