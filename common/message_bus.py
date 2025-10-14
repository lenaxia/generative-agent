"""Event-driven communication system for the StrandsAgent Universal Agent System.

Provides message bus functionality for inter-component communication,
event handling, and workflow coordination across the system.
"""

import asyncio
import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from common.enhanced_event_context import (
    LLMSafeEventContext,
    create_context_from_event_data,
)
from common.intent_processor import IntentProcessor
from common.intents import Intent

logger = logging.getLogger("supervisor")


class MessageType(Enum):
    """Enumeration of message types for inter-component communication.

    Defines the different types of messages that can be published and
    subscribed to within the message bus system.

    NOTE: This enum is maintained for backward compatibility.
    New event types should be registered dynamically via MessageTypeRegistry.
    """

    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESPONSE = "task_response"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    AGENT_STATUS = "agent_status"
    AGENT_EVENT = "agent_event"
    AGENT_ERROR = "agent_error"
    SEND_MESSAGE = "send_message"
    INCOMING_REQUEST = "incoming_request"
    TIMER_EXPIRED = "timer_expired"
    AGENT_QUESTION = "agent_question"
    USER_RESPONSE = "user_response"


class EventSchema:
    """Schema definition for event data validation."""

    def __init__(self, event_type: str, schema: dict[str, Any], description: str = ""):
        """Initialize event schema.

        Args:
            event_type: The event type name
            schema: Schema definition with field requirements
            description: Human-readable description of the event
        """
        self.event_type = event_type
        self.schema = schema
        self.description = description

    def validate(self, data: dict[str, Any]) -> bool:
        """Validate event data against schema.

        Args:
            data: Event data to validate

        Returns:
            True if data is valid, False otherwise
        """
        # Basic validation - could be enhanced with jsonschema
        required_fields = [
            k
            for k, v in self.schema.items()
            if isinstance(v, dict) and v.get("required", False)
        ]
        return all(field in data for field in required_fields)


class MessageTypeRegistry:
    """Dynamic registry for message types with schema validation."""

    def __init__(self):
        """Initialize the registry with core system events."""
        self._registered_types: set[str] = set()
        self._schemas: dict[str, EventSchema] = {}
        self._publishers: dict[str, list[str]] = {}
        self._subscribers: dict[str, list[str]] = {}

        # Register core system events
        self._register_core_events()

    def _register_core_events(self):
        """Register essential system events."""
        core_events = [
            "WORKFLOW_STARTED",
            "WORKFLOW_COMPLETED",
            "WORKFLOW_FAILED",
            "TASK_STARTED",
            "TASK_COMPLETED",
            "TASK_FAILED",
            "AGENT_ROLE_SWITCHED",
            "SYSTEM_HEALTH_CHECK",
            "HEARTBEAT_TICK",
            "FAST_HEARTBEAT_TICK",
        ]
        for event in core_events:
            self.register_event_type(event, "system", {})

    def register_event_type(
        self,
        event_type: str,
        publisher_role: str,
        schema: Optional[dict[str, Any]] = None,
        description: str = "",
    ):
        """Register a new event type from a role.

        Args:
            event_type: Name of the event type
            publisher_role: Role that publishes this event
            schema: Optional schema for validation
            description: Human-readable description
        """
        self._registered_types.add(event_type)

        if schema:
            self._schemas[event_type] = EventSchema(event_type, schema, description)

        if event_type not in self._publishers:
            self._publishers[event_type] = []
        if publisher_role not in self._publishers[event_type]:
            self._publishers[event_type].append(publisher_role)

        logger.info(f"Registered event '{event_type}' from role '{publisher_role}'")

    def register_subscription(self, event_type: str, subscriber_role: str):
        """Register a role's subscription to an event.

        Args:
            event_type: Event type to subscribe to
            subscriber_role: Role that subscribes to this event
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if subscriber_role not in self._subscribers[event_type]:
            self._subscribers[event_type].append(subscriber_role)

        logger.info(f"Role '{subscriber_role}' subscribed to '{event_type}'")

    def is_valid_event_type(self, event_type: str) -> bool:
        """Check if an event type is registered.

        Args:
            event_type: Event type to check

        Returns:
            True if event type is registered
        """
        return event_type in self._registered_types

    def validate_event_data(self, event_type: str, data: dict[str, Any]) -> bool:
        """Validate event data against registered schema.

        Args:
            event_type: Event type to validate
            data: Event data to validate

        Returns:
            True if data is valid or no schema exists
        """
        if event_type in self._schemas:
            return self._schemas[event_type].validate(data)
        return True  # No schema = no validation required

    def get_event_documentation(self) -> dict[str, Any]:
        """Get complete event system documentation.

        Returns:
            Dictionary with event system information
        """
        return {
            "registered_events": list(self._registered_types),
            "publishers": self._publishers,
            "subscribers": self._subscribers,
            "schemas": {k: v.schema for k, v in self._schemas.items()},
        }


class MessageBus:
    """Thread-safe message bus for event-driven communication.

    Provides publish-subscribe functionality for inter-component communication
    within the StrandsAgent Universal Agent System. Supports multiple subscribers
    per message type and executes callbacks in separate threads for non-blocking
    message processing.
    """

    class Config:
        """Configuration class for MessageBus.

        Allows arbitrary types to be used in the message bus system.
        """

        arbitrary_types_allowed = True

    def __init__(self):
        """Initialize the MessageBus with empty subscriber registry.

        Sets up the internal data structures for managing subscribers,
        thread safety lock, running state, and dynamic event registry.
        """
        # Support both MessageType enum and string-based dynamic events
        self._subscribers: dict[str, dict[Any, list[Callable]]] = {}
        # REMOVED: threading.Lock() - no longer needed with single event loop
        self._running = False
        self.event_registry = MessageTypeRegistry()

        # NEW: Intent processing for LLM-safe architecture
        self._intent_processor: Optional[IntentProcessor] = None
        self._enable_intent_processing = True

        # Dependencies for intent processor (set by supervisor)
        self.communication_manager = None
        self.workflow_engine = None
        self.llm_factory = None

    def start(self):
        """Start the message bus to begin processing messages.

        Sets the running state to True, allowing the message bus to
        process and deliver published messages to subscribers.
        """
        self._running = True
        logger.info("MessageBus started in LLM-safe mode")

        # Initialize intent processor when dependencies are available
        if self._enable_intent_processing:
            self._initialize_intent_processor()

    def stop(self):
        """Stop the message bus from processing messages.

        Sets the running state to False, preventing the message bus from
        processing any new published messages until started again.
        """
        self._running = False

    def is_running(self):
        """Check if the message bus is currently running.

        Returns:
            bool: True if the message bus is running and processing messages,
                False otherwise.
        """
        return self._running

    def publish(self, publisher, message_type, message: Any):
        """Publish a message to all subscribers of the specified message type.

        Delivers the message to all registered subscribers for the given message type.
        Uses the event loop for all callbacks to maintain single event loop architecture.
        Supports both MessageType enum (for backward compatibility) and string-based
        dynamic event types.

        Args:
            publisher: The entity publishing the message (for logging/tracking).
            message_type: The type of message being published (MessageType enum or string).
            message: The message content to be delivered to subscribers.
        """
        # Convert MessageType enum to string for unified handling
        if isinstance(message_type, MessageType):
            event_type = message_type.value
        else:
            event_type = message_type

        # Validate dynamic event types
        if not isinstance(message_type, MessageType):
            if not self.event_registry.is_valid_event_type(event_type):
                logger.warning(
                    f"Publishing unknown event type '{event_type}' from {publisher}"
                )
                # Still allow it - might be a new event type not yet registered

            # Validate event data if schema exists
            if isinstance(
                message, dict
            ) and not self.event_registry.validate_event_data(event_type, message):
                logger.warning(
                    f"Event data validation failed for '{event_type}': {message}"
                )
        if not self.is_running():
            return

        # LLM-SAFE: No longer need threading lock with single event loop
        if event_type not in self._subscribers:
            return

        logger.debug(f"Publishing message: [{event_type}] {message}")

        # Create a copy of the subscribers to avoid modifying the dictionary while iterating
        subscribers_copy = self._subscribers[event_type].copy()

        # Check if we have a running event loop
        try:
            loop = asyncio.get_running_loop()
            has_running_loop = True
        except RuntimeError:
            has_running_loop = False
            loop = None

        # Process all callbacks
        for _subscriber, callbacks in subscribers_copy.items():
            for callback in callbacks:
                callback_name = getattr(callback, "__name__", "unknown_callback")

                # Handle async callbacks
                if asyncio.iscoroutinefunction(callback):
                    if has_running_loop:
                        # Schedule coroutine in the running event loop
                        try:
                            asyncio.create_task(
                                self._run_async_callback_in_loop(
                                    callback, message, callback_name
                                )
                            )
                            logger.debug(
                                f"Scheduled async callback {callback_name} in event loop"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error scheduling async callback {callback_name}: {e}"
                            )
                    else:
                        # No event loop running, run the callback synchronously
                        # This is needed for tests and synchronous environments
                        try:
                            logger.debug(
                                f"Running async callback {callback_name} synchronously"
                            )
                            # Create a temporary event loop to run the async callback
                            temp_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(temp_loop)
                            try:
                                temp_loop.run_until_complete(
                                    self._run_async_callback_in_loop(
                                        callback, message, callback_name
                                    )
                                )
                            finally:
                                temp_loop.close()
                                asyncio.set_event_loop(None)
                        except Exception as e:
                            logger.error(
                                f"Error running async callback {callback_name} synchronously: {e}"
                            )
                else:
                    # For sync callbacks
                    if has_running_loop:
                        # Schedule in the running event loop
                        try:
                            # Create a wrapper function to avoid None type issues
                            def run_callback_wrapper():
                                self._run_sync_callback_in_loop(
                                    callback, message, callback_name, publisher
                                )

                            loop.call_soon(run_callback_wrapper)
                            logger.debug(
                                f"Scheduled sync callback {callback_name} in event loop"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error scheduling sync callback {callback_name}: {e}"
                            )
                    else:
                        # No event loop, run directly
                        try:
                            self._run_sync_callback_in_loop(
                                callback, message, callback_name, publisher
                            )
                        except Exception as e:
                            logger.error(
                                f"Error running sync callback {callback_name}: {e}"
                            )

    async def _run_async_callback_in_loop(
        self, callback: Callable, message: Any, callback_name: str
    ):
        """Run async callback in the current event loop with error handling."""
        try:
            logger.debug(f"Running async callback {callback_name} in event loop")
            result = await callback(message)
            logger.debug(f"Async callback {callback_name} completed successfully")

            # Process intents returned by the callback (LLM-safe architecture)
            await self._process_callback_intents(result, callback_name)
        except Exception as e:
            logger.error(f"Error in async callback {callback_name}: {e}")

    def _run_sync_callback_in_loop(
        self, callback: Callable, message: Any, callback_name: str, publisher=None
    ):
        """Run sync callback in the event loop with error handling."""
        try:
            logger.debug(f"Running sync callback {callback_name}")

            # Create context for handlers that expect it
            context = self._create_event_context(publisher, message)

            # Try calling with context first, fallback to message only
            try:
                result = callback(message, context)
            except TypeError:
                # Handler doesn't expect context, call with message only
                result = callback(message)

            logger.debug(f"Sync callback {callback_name} completed successfully")

            # Process intents returned by the callback (LLM-safe architecture)
            # Schedule async processing for sync callbacks
            if result is not None:
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(
                        self._process_callback_intents(result, callback_name)
                    )
                except RuntimeError:
                    # No running event loop, process synchronously
                    asyncio.run(self._process_callback_intents(result, callback_name))
        except Exception as e:
            logger.error(f"Error in sync callback {callback_name}: {e}")

    async def _process_callback_intents(self, result: Any, callback_name: str):
        """Process intents returned by event handlers (LLM-safe architecture)."""
        if result is None:
            return

        # Convert single intent to list
        if hasattr(result, "validate") and callable(getattr(result, "validate")):
            intents = [result]
        elif isinstance(result, list):
            intents = result
        else:
            # Not an intent, ignore
            return

        # Filter valid intents
        valid_intents = []
        for intent in intents:
            if hasattr(intent, "validate") and callable(getattr(intent, "validate")):
                try:
                    if intent.validate():
                        valid_intents.append(intent)
                    else:
                        logger.warning(f"Invalid intent from {callback_name}: {intent}")
                except Exception as e:
                    logger.error(f"Error validating intent from {callback_name}: {e}")
            else:
                logger.debug(f"Non-intent result from {callback_name}: {type(result)}")

        # Process valid intents through intent processor
        if valid_intents and self._intent_processor:
            try:
                logger.debug(
                    f"Processing {len(valid_intents)} intents from {callback_name}"
                )
                await self._intent_processor.process_intents(valid_intents)
            except Exception as e:
                logger.error(f"Error processing intents from {callback_name}: {e}")

    def subscribe(self, subscriber, message_type, callback: Callable):
        """Subscribe to messages of a specific type.

        Registers a callback function to be executed when messages of the
        specified type are published. Multiple callbacks can be registered
        for the same subscriber and message type.
        Supports both MessageType enum and string-based dynamic event types.

        Args:
            subscriber: The entity subscribing to messages (for tracking).
            message_type: The type of messages to subscribe to (MessageType enum or string).
            callback: The function to call when a message is received.
                     Must accept one argument (the message).
        """
        # Convert MessageType enum to string for unified handling
        if isinstance(message_type, MessageType):
            event_type = message_type.value
        else:
            event_type = message_type

        # LLM-SAFE: No longer need threading lock with single event loop
        if event_type not in self._subscribers:
            self._subscribers[event_type] = {}

        if subscriber not in self._subscribers[event_type]:
            self._subscribers[event_type][subscriber] = []

        self._subscribers[event_type][subscriber].append(callback)

        logger.info(f"Subscribed {subscriber} to event '{event_type}'")

    def unsubscribe(
        self, subscriber, message_type, callback: Optional[Callable] = None
    ):
        """Unsubscribe from messages of a specific type.

        Removes a subscriber's callback(s) for the specified message type.
        If no callback is specified, all callbacks for the subscriber and
        message type are removed.
        Supports both MessageType enum and string-based dynamic event types.

        Args:
            subscriber: The entity unsubscribing from messages.
            message_type: The type of messages to unsubscribe from (MessageType enum or string).
            callback: The specific callback to remove. If None, all callbacks
                     for the subscriber and message type are removed.
        """
        # Convert MessageType enum to string for unified handling
        if isinstance(message_type, MessageType):
            event_type = message_type.value
        else:
            event_type = message_type

            # LLM-SAFE: No longer need threading lock with single event loop
            if event_type not in self._subscribers:
                return

            if subscriber in self._subscribers[event_type]:
                if callback is None:
                    del self._subscribers[event_type][subscriber]
                else:
                    self._subscribers[event_type][subscriber] = [
                        cb
                        for cb in self._subscribers[event_type][subscriber]
                        if cb != callback
                    ]

    def _initialize_intent_processor(self):
        """Initialize intent processor with available dependencies."""
        if not self._intent_processor and self._enable_intent_processing:
            self._intent_processor = IntentProcessor(
                communication_manager=self.communication_manager,
                workflow_engine=self.workflow_engine,
            )
            logger.info("Intent processor initialized")

            # Connect IntentProcessor to RoleRegistry for intent handler registration
            if hasattr(self, "workflow_engine") and self.workflow_engine:
                if (
                    hasattr(self.workflow_engine, "role_registry")
                    and self.workflow_engine.role_registry
                ):
                    self.workflow_engine.role_registry.set_intent_processor(
                        self._intent_processor
                    )
                    logger.info("IntentProcessor connected to RoleRegistry")

    def set_dependencies(
        self, communication_manager=None, workflow_engine=None, llm_factory=None
    ):
        """Set dependencies for intent processing."""
        if communication_manager:
            self.communication_manager = communication_manager
        if workflow_engine:
            self.workflow_engine = workflow_engine
        if llm_factory:
            self.llm_factory = llm_factory

        # Re-initialize intent processor with new dependencies
        if self._running and self._enable_intent_processing:
            self._initialize_intent_processor()

    async def publish_async(self, publisher, event_type: str, message: Any):
        """
        LLM-SAFE: Enhanced publish with intent processing support.

        This method supports pure function event handlers that return intents,
        enabling thread-safe event processing without complex async operations.

        Args:
            publisher: The entity publishing the event
            event_type: Type of event being published
            message: Event data payload
        """
        if not self._running:
            return

        # Create explicit context for handlers
        context = self._create_event_context(publisher, message)

        # Process subscribers
        if event_type in self._subscribers:
            for role_name, handlers in self._subscribers[event_type].items():
                for handler in handlers:
                    try:
                        # Call handler with appropriate parameters
                        import inspect

                        sig = inspect.signature(handler)
                        param_count = len(sig.parameters)

                        if asyncio.iscoroutinefunction(handler):
                            if param_count >= 2:
                                result = await handler(message, context)
                            else:
                                result = await handler(message)
                        else:
                            if param_count >= 2:
                                result = handler(message, context)
                            else:
                                result = handler(message)

                        # Process intents if returned
                        if self._is_intent_list(result):
                            await self._process_intents(result)

                    except Exception as e:
                        logger.error(f"Handler error in {role_name}: {e}")

    def _create_event_context(self, publisher, message) -> LLMSafeEventContext:
        """Create explicit event context for handlers."""
        # Extract publisher information
        user_id = getattr(publisher, "user_id", None)
        channel_id = getattr(publisher, "channel_id", None)
        source = publisher.__class__.__name__ if publisher else "unknown"

        # Create context from event data
        return create_context_from_event_data(
            event_data=message, source=source, user_id=user_id, channel_id=channel_id
        )

    def _is_intent_list(self, result) -> bool:
        """Check if result is list of intents."""
        return (
            isinstance(result, list)
            and len(result) > 0
            and all(isinstance(item, Intent) for item in result)
        )

    async def _process_intents(self, intents: list[Intent]):
        """Process intents using intent processor."""
        if self._intent_processor:
            await self._intent_processor.process_intents(intents)
        else:
            logger.warning("No intent processor available - intents not processed")

    def get_intent_processor_metrics(self) -> dict[str, Any]:
        """Get metrics from intent processor."""
        if self._intent_processor:
            return {
                "processed_count": self._intent_processor.get_processed_count(),
                "registered_handlers": self._intent_processor.get_registered_handlers(),
            }
        return {"processed_count": 0, "registered_handlers": {}}

    def enable_intent_processing(self, enable: bool = True):
        """Enable or disable intent processing."""
        self._enable_intent_processing = enable
        if enable and self._running:
            self._initialize_intent_processor()
        elif not enable:
            self._intent_processor = None
        logger.info(f"Intent processing {'enabled' if enable else 'disabled'}")
