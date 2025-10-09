"""Event-driven communication system for the StrandsAgent Universal Agent System.

Provides message bus functionality for inter-component communication,
event handling, and workflow coordination across the system.
"""

import logging
import threading
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("supervisor")


class MessageType(Enum):
    """Enumeration of message types for inter-component communication.

    Defines the different types of messages that can be published and
    subscribed to within the message bus system.
    """

    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESPONSE = "task_response"
    AGENT_STATUS = "agent_status"
    AGENT_EVENT = "agent_event"
    AGENT_ERROR = "agent_error"
    SEND_MESSAGE = "send_message"
    INCOMING_REQUEST = "incoming_request"
    TIMER_EXPIRED = "timer_expired"


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
        thread safety lock, and running state.
        """
        self._subscribers: dict[MessageType, dict[Any, list[Callable]]] = {}
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        """Start the message bus to begin processing messages.

        Sets the running state to True, allowing the message bus to
        process and deliver published messages to subscribers.
        """
        self._running = True

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

    def publish(self, publisher, message_type: MessageType, message: Any):
        """Publish a message to all subscribers of the specified message type.

        Delivers the message to all registered subscribers for the given message type.
        Each callback is executed in a separate thread for non-blocking processing.

        Args:
            publisher: The entity publishing the message (for logging/tracking).
            message_type: The type of message being published.
            message: The message content to be delivered to subscribers.
        """
        if not self.is_running():
            return

        with self._lock:
            if message_type not in self._subscribers:
                return

            logger.info(f"Publishing message: [{message_type}] {message}")

            # Create a copy of the subscribers to avoid modifying the dictionary while iterating
            subscribers_copy = self._subscribers[message_type].copy()

        # Release the lock before executing the callbacks
        for _subscriber, callbacks in subscribers_copy.items():
            for callback in callbacks:
                # Start a new thread for each callback
                callback_thread = threading.Thread(target=callback, args=(message,))
                callback_thread.start()

    def subscribe(self, subscriber, message_type: MessageType, callback: Callable):
        """Subscribe to messages of a specific type.

        Registers a callback function to be executed when messages of the
        specified type are published. Multiple callbacks can be registered
        for the same subscriber and message type.

        Args:
            subscriber: The entity subscribing to messages (for tracking).
            message_type: The type of messages to subscribe to.
            callback: The function to call when a message is received.
                     Must accept one argument (the message).
        """
        with self._lock:
            if message_type not in self._subscribers:
                self._subscribers[message_type] = {}

            if subscriber not in self._subscribers[message_type]:
                self._subscribers[message_type][subscriber] = []

            self._subscribers[message_type][subscriber].append(callback)

    def unsubscribe(
        self, subscriber, message_type: MessageType, callback: Callable = None
    ):
        """Unsubscribe from messages of a specific type.

        Removes a subscriber's callback(s) for the specified message type.
        If no callback is specified, all callbacks for the subscriber and
        message type are removed.

        Args:
            subscriber: The entity unsubscribing from messages.
            message_type: The type of messages to unsubscribe from.
            callback: The specific callback to remove. If None, all callbacks
                     for the subscriber and message type are removed.
        """
        with self._lock:
            if message_type not in self._subscribers:
                return

            if subscriber in self._subscribers[message_type]:
                if callback is None:
                    del self._subscribers[message_type][subscriber]
                else:
                    self._subscribers[message_type][subscriber] = [
                        cb
                        for cb in self._subscribers[message_type][subscriber]
                        if cb != callback
                    ]
