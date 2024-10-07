from enum import Enum
from typing import Any, Callable, Dict, List
import threading
from pydantic import BaseModel
import logging

logger = logging.getLogger("supervisor")

class MessageType(Enum):
    TASK_ASSIGNMENT = 'task_assignment'
    TASK_RESPONSE = 'task_response'
    AGENT_STATUS = 'agent_status'
    AGENT_EVENT = 'agent_event'
    AGENT_ERROR = 'agent_error'

class MessageBus:
    class Config:
        arbitrary_types_allowed = True

    def __init__(self):
        self._subscribers: Dict[MessageType, Dict[Any, List[Callable]]] = {}
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def is_running(self):
        return self._running

    def publish(self, publisher, message_type: MessageType, message: Any):
        if not self.is_running():
            return

        with self._lock:
            if message_type not in self._subscribers:
                return

            logger.info(f"Publishing message: {message}")

            # Create a copy of the subscribers to avoid modifying the dictionary while iterating
            subscribers_copy = self._subscribers[message_type].copy()

        # Release the lock before executing the callbacks
        for subscriber, callbacks in subscribers_copy.items():
            for callback in callbacks:
                # Start a new thread for each callback
                callback_thread = threading.Thread(target=callback, args=(message,))
                callback_thread.start()

    def subscribe(self, subscriber, message_type: MessageType, callback: Callable):
        with self._lock:
            if message_type not in self._subscribers:
                self._subscribers[message_type] = {}

            if subscriber not in self._subscribers[message_type]:
                self._subscribers[message_type][subscriber] = []

            self._subscribers[message_type][subscriber].append(callback)

    def unsubscribe(self, subscriber, message_type: MessageType, callback: Callable = None):
        with self._lock:
            if message_type not in self._subscribers:
                return

            if subscriber in self._subscribers[message_type]:
                if callback is None:
                    del self._subscribers[message_type][subscriber]
                else:
                    self._subscribers[message_type][subscriber] = [
                        cb for cb in self._subscribers[message_type][subscriber] if cb != callback
                    ]
