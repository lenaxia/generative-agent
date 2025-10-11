"""Event-driven communication system for the StrandsAgent Universal Agent System.

Provides message bus functionality for inter-component communication,
event handling, and workflow coordination across the system.
"""

import asyncio
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
    AGENT_QUESTION = "agent_question"
    USER_RESPONSE = "user_response"


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
                # Check if callback is async
                if asyncio.iscoroutinefunction(callback):
                    # Schedule async callback in the event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # Use call_soon_threadsafe for better cross-thread safety
                        future = asyncio.run_coroutine_threadsafe(
                            callback(message), loop
                        )
                        logger.debug(
                            f"Scheduled async callback {callback.__name__} in event loop"
                        )
                    except RuntimeError as e:
                        # No running event loop, start a new thread with event loop
                        logger.debug(
                            f"No running event loop for {callback.__name__}, creating new thread: {e}"
                        )
                        callback_thread = threading.Thread(
                            target=self._run_async_callback_safely,
                            args=(callback, message),
                            name=f"async_callback_{callback.__name__}",
                        )
                        callback_thread.daemon = True
                        callback_thread.start()
                    except Exception as e:
                        logger.error(
                            f"Error scheduling async callback {callback.__name__}: {e}"
                        )
                else:
                    # Start a new thread for sync callback
                    callback_thread = threading.Thread(
                        target=self._run_sync_callback_safely,
                        args=(callback, message),
                        name=f"sync_callback_{callback.__name__}",
                    )
                    callback_thread.daemon = True
                    callback_thread.start()

    def _run_async_callback_safely(self, callback: Callable, message: Any):
        """Run async callback in a new event loop with error handling."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.debug(
                f"Running async callback {callback.__name__} in new event loop"
            )
            loop.run_until_complete(callback(message))
            logger.debug(f"Async callback {callback.__name__} completed successfully")
        except Exception as e:
            logger.error(f"Error in async callback {callback.__name__}: {e}")
        finally:
            loop.close()

    def _run_sync_callback_safely(self, callback: Callable, message: Any):
        """Run sync callback with error handling."""
        try:
            logger.debug(f"Running sync callback {callback.__name__}")
            callback(message)
            logger.debug(f"Sync callback {callback.__name__} completed successfully")
        except Exception as e:
            logger.error(f"Error in sync callback {callback.__name__}: {e}")

    def _run_async_callback(self, callback: Callable, message: Any):
        """Run async callback in a new event loop (deprecated - use _run_async_callback_safely)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(callback(message))
        finally:
            loop.close()

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
