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
from typing import Any, Optional


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
    user_id: str | None = None
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
    user_id: str | None = None
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

    This intent supports two modes:
    1. Simple workflow: Just workflow_type + parameters (generic workflow request)
    2. Explicit task graph: Includes tasks + dependencies (detailed execution plan)

    The workflow engine handles both modes, converting to TaskGraph internally.
    """

    workflow_type: str
    parameters: dict[str, Any]
    priority: int = 1
    context: dict[str, Any] | None = None

    # Optional: Explicit task graph for detailed workflow execution
    tasks: list[dict[str, Any]] | None = None
    dependencies: list[dict[str, Any]] | None = None
    request_id: str | None = None
    user_id: str | None = None
    channel_id: str | None = None
    original_instruction: str | None = None

    def validate(self) -> bool:
        """Validate workflow intent parameters."""
        # Basic validation for simple workflow
        basic_valid = (
            bool(self.workflow_type and isinstance(self.parameters, dict))
            and isinstance(self.priority, int)
            and self.priority >= 0
            and len(self.workflow_type.strip()) > 0
        )

        # If tasks are provided, validate task graph structure
        if self.tasks is not None:
            task_graph_valid = (
                bool(self.tasks)
                and bool(self.request_id)
                and all(task.get("id") and task.get("role") for task in self.tasks)
            )
            return bool(basic_valid and task_graph_valid)

        return basic_valid

    def is_task_graph_workflow(self) -> bool:
        """Check if this is an explicit task graph workflow.

        Returns True if tasks field is set (even if empty), indicating this is
        a task graph workflow rather than a simple workflow type request.
        """
        return self.tasks is not None

    def get_expected_workflow_ids(self) -> set[str]:
        """Get set of workflow IDs this intent will spawn (for task graph workflows).

        Returns:
            Set of workflow IDs that will be created from this intent's tasks.
        """
        if not self.is_task_graph_workflow() or not self.tasks or not self.request_id:
            return set()
        return {f"{self.request_id}_task_{task['id']}" for task in self.tasks}

    def to_dict(self) -> dict[str, Any]:
        """Convert intent to dictionary for serialization."""
        result = {
            "workflow_type": self.workflow_type,
            "parameters": self.parameters,
            "priority": self.priority,
        }
        if self.context:
            result["context"] = self.context
        if self.is_task_graph_workflow():
            result.update(
                {
                    "tasks": self.tasks,
                    "dependencies": self.dependencies,
                    "request_id": self.request_id,
                    "user_id": self.user_id,
                    "channel_id": self.channel_id,
                    "original_instruction": self.original_instruction,
                }
            )
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowIntent":
        """Create WorkflowIntent from dictionary."""
        return cls(
            workflow_type=data["workflow_type"],
            parameters=data["parameters"],
            priority=data.get("priority", 1),
            context=data.get("context"),
            tasks=data.get("tasks"),
            dependencies=data.get("dependencies"),
            request_id=data.get("request_id"),
            user_id=data.get("user_id"),
            channel_id=data.get("channel_id"),
            original_instruction=data.get("original_instruction"),
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
    user_id: str | None = None

    def validate(self) -> bool:
        """Validate error intent parameters."""
        return (
            bool(self.error_type and self.error_message)
            and isinstance(self.error_details, dict)
            and isinstance(self.recoverable, bool)
            and len(self.error_type.strip()) > 0
            and len(self.error_message.strip()) > 0
        )


@dataclass
class MemoryWriteIntent(Intent):
    """
    Universal intent: Any role can write memories.

    This intent represents the desire to store a memory in the unified memory system.
    The actual memory storage is handled by the infrastructure asynchronously.
    """

    user_id: str
    memory_type: str  # conversation, event, plan, preference, fact
    content: str
    source_role: str
    importance: float = 0.5
    metadata: dict[str, Any] | None = None
    tags: list[str] | None = None
    related_memories: list[str] | None = None

    def validate(self) -> bool:
        """Validate memory write intent parameters."""
        valid_types = ["conversation", "event", "plan", "preference", "fact"]
        return (
            bool(
                self.user_id and self.memory_type and self.content and self.source_role
            )
            and self.memory_type in valid_types
            and 0.0 <= self.importance <= 1.0
            and len(self.user_id.strip()) > 0
            and len(self.content.strip()) > 0
            and len(self.source_role.strip()) > 0
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
    error: Exception, context: dict[str, Any] | None = None
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
