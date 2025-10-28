"""WorkflowExecutionIntent for event-driven TaskGraph execution.

This module defines the intent structure for executing multi-step workflows
through the event-driven architecture described in Document 34 Phase 2.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from common.intents import Intent


@dataclass
class WorkflowExecutionIntent(Intent):
    """Intent for executing multi-step workflows."""

    tasks: list[dict[str, Any]]
    dependencies: list[dict[str, Any]]
    request_id: str
    user_id: str
    channel_id: str
    original_instruction: str
    created_at: float = 0.0

    def __post_init__(self):
        """Set creation timestamp."""
        if self.created_at is None:
            self.created_at = time.time()

    def validate(self) -> bool:
        """Validate workflow execution intent."""
        return bool(
            self.tasks
            and self.request_id
            and all(task.get("id") and task.get("role") for task in self.tasks)
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert intent to dictionary for serialization."""
        return {
            "type": "workflow_execution",
            "tasks": self.tasks,
            "dependencies": self.dependencies,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "original_instruction": self.original_instruction,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowExecutionIntent":
        """Create intent from dictionary."""
        return cls(
            tasks=data["tasks"],
            dependencies=data["dependencies"],
            request_id=data["request_id"],
            user_id=data["user_id"],
            channel_id=data["channel_id"],
            original_instruction=data["original_instruction"],
            created_at=data.get("created_at", 0.0),
        )

    def get_expected_workflow_ids(self) -> set[str]:
        """Get set of workflow IDs this intent will spawn.

        Following Document 35 design for workflow lifecycle tracking.

        Returns:
            Set of workflow IDs that will be created from this intent's tasks.
        """
        return {f"{self.request_id}_task_{task['id']}" for task in self.tasks}
