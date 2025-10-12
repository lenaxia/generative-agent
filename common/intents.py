"""
LLM-SAFE Intent System Foundation

This module provides the core intent system for declarative event processing.
All intents represent "what should happen" rather than "how to make it happen",
enabling pure functional event handlers and eliminating threading complexity.

Created: 2025-10-12
Part of: Threading Architecture Improvements (Documents 25, 26, 27)
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Intent(ABC):
    """
    LLM-SAFE: Base class for all declarative intents.

    All intents must:
    1. Be immutable data structures (dataclass)
    2. Implement validation logic
    3. Represent declarative actions, not imperative operations
    4. Be serializable for debugging and monitoring
    """

    def __post_init__(self):
        """Auto-set creation timestamp if not provided."""
        if not hasattr(self, "created_at") or self.created_at is None:
            self.created_at = time.time()

    @abstractmethod
    def validate(self) -> bool:
        """
        All intents must implement validation.

        Returns:
            bool: True if intent is valid, False otherwise
        """
        pass

    def to_dict(self) -> dict[str, Any]:
        """Convert intent to dictionary for serialization."""
        return {
            "type": self.__class__.__name__,
            "data": self.__dict__,
            "created_at": self.created_at,
        }


@dataclass
class NotificationIntent(Intent):
    """
    Universal intent: Any role can send notifications.

    This intent represents the desire to send a notification to a user or channel.
    The actual notification delivery is handled by the infrastructure.
    """

    message: str
    channel: str
    user_id: Optional[str] = None
    priority: str = "medium"  # "low", "medium", "high"
    notification_type: str = "info"  # "info", "warning", "error", "success"

    def validate(self) -> bool:
        """Validate notification intent parameters."""
        return (
            bool(self.message and self.channel)
            and self.priority in ["low", "medium", "high"]
            and self.notification_type in ["info", "warning", "error", "success"]
            and len(self.message.strip()) > 0
        )


@dataclass
class AuditIntent(Intent):
    """
    Universal intent: Any role can audit actions.

    This intent represents the desire to log an action for audit purposes.
    The actual audit logging is handled by the infrastructure.
    """

    action: str
    details: dict[str, Any]
    user_id: Optional[str] = None
    severity: str = "info"  # "debug", "info", "warning", "error", "critical"

    def validate(self) -> bool:
        """Validate audit intent parameters."""
        return (
            bool(self.action and isinstance(self.details, dict))
            and self.severity in ["debug", "info", "warning", "error", "critical"]
            and len(self.action.strip()) > 0
        )


@dataclass
class WorkflowIntent(Intent):
    """
    Universal intent: Any role can start workflows.

    This intent represents the desire to start a new workflow.
    The actual workflow execution is handled by the infrastructure.
    """

    workflow_type: str
    parameters: dict[str, Any]
    priority: int = 1
    context: Optional[dict[str, Any]] = None

    def validate(self) -> bool:
        """Validate workflow intent parameters."""
        return (
            bool(self.workflow_type and isinstance(self.parameters, dict))
            and isinstance(self.priority, int)
            and self.priority >= 0
            and len(self.workflow_type.strip()) > 0
        )


@dataclass
class ErrorIntent(Intent):
    """
    Universal intent: Any role can report errors.

    This intent represents the occurrence of an error that needs to be handled
    by the infrastructure (logging, alerting, recovery, etc.).
    """

    error_type: str
    error_message: str
    error_details: dict[str, Any]
    recoverable: bool = True
    user_id: Optional[str] = None

    def validate(self) -> bool:
        """Validate error intent parameters."""
        return (
            bool(self.error_type and self.error_message)
            and isinstance(self.error_details, dict)
            and isinstance(self.recoverable, bool)
            and len(self.error_type.strip()) > 0
            and len(self.error_message.strip()) > 0
        )


# Utility functions for intent handling
def validate_intent_list(intents: list) -> bool:
    """
    Validate a list of intents.

    Args:
        intents: List of intent objects

    Returns:
        bool: True if all intents are valid, False otherwise
    """
    if not isinstance(intents, list):
        return False

    return all(isinstance(intent, Intent) and intent.validate() for intent in intents)


def create_error_intent(
    error: Exception, context: Optional[dict[str, Any]] = None
) -> ErrorIntent:
    """
    Create an ErrorIntent from an exception.

    Args:
        error: Exception that occurred
        context: Optional context information

    Returns:
        ErrorIntent: Intent representing the error
    """
    return ErrorIntent(
        error_type=error.__class__.__name__,
        error_message=str(error),
        error_details={
            "context": context or {},
            "exception_args": error.args if hasattr(error, "args") else [],
        },
        recoverable=not isinstance(error, (SystemExit, KeyboardInterrupt)),
    )


def create_notification_from_error(
    error: Exception, channel: str = "general"
) -> NotificationIntent:
    """
    Create a NotificationIntent from an exception.

    Args:
        error: Exception that occurred
        channel: Channel to send notification to

    Returns:
        NotificationIntent: Intent to notify about the error
    """
    return NotificationIntent(
        message=f"Error occurred: {error.__class__.__name__}: {str(error)}",
        channel=channel,
        priority="high",
        notification_type="error",
    )
