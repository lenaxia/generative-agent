"""
Workflow Duration Logger

Tracks and logs workflow execution durations for performance monitoring and analytics.
Supports both CLI and Slack workflow completion tracking.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class WorkflowSource(str, Enum):
    """Source of workflow execution."""

    CLI = "CLI"
    SLACK = "SLACK"
    API = "API"
    UNKNOWN = "UNKNOWN"


class WorkflowType(str, Enum):
    """Type of workflow execution."""

    FAST_REPLY = "FAST_REPLY"
    COMPLEX_WORKFLOW = "COMPLEX_WORKFLOW"
    UNKNOWN = "UNKNOWN"


@dataclass
class WorkflowDurationMetrics:
    """Metrics for workflow duration tracking."""

    workflow_id: str
    source: WorkflowSource
    workflow_type: WorkflowType
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    instruction: Optional[str] = None
    role: Optional[str] = None
    confidence: Optional[float] = None
    task_count: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.fromtimestamp(self.start_time).isoformat()

        if self.end_time and self.duration_seconds is None:
            self.duration_seconds = self.end_time - self.start_time

    def complete(
        self,
        end_time: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        """Mark the workflow as completed and calculate duration."""
        self.end_time = end_time or time.time()
        self.duration_seconds = self.end_time - self.start_time
        self.success = success
        if error_message:
            self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class WorkflowDurationLogger:
    """
    Centralized workflow duration logging system.

    Tracks workflow execution times across CLI and Slack interfaces,
    providing performance metrics and analytics capabilities.
    """

    def __init__(
        self,
        log_file_path: str = "logs/workflow_durations.jsonl",
        enable_console_logging: bool = True,
        max_log_file_size_mb: int = 100,
    ):
        """
        Initialize the workflow duration logger.

        Args:
            log_file_path: Path to the JSONL log file for storing duration metrics
            enable_console_logging: Whether to also log to console
            max_log_file_size_mb: Maximum log file size before rotation
        """
        self.log_file_path = log_file_path
        self.enable_console_logging = enable_console_logging
        self.max_log_file_size_bytes = max_log_file_size_mb * 1024 * 1024

        # Active workflow tracking
        self.active_workflows: Dict[str, WorkflowDurationMetrics] = {}

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.WorkflowDurationLogger")

        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

        # Rotate log file if it's too large
        self._rotate_log_file_if_needed()

        self.logger.info(
            f"WorkflowDurationLogger initialized with log file: {self.log_file_path}"
        )

    def start_workflow_tracking(
        self,
        workflow_id: str,
        source: WorkflowSource,
        workflow_type: WorkflowType = WorkflowType.UNKNOWN,
        instruction: Optional[str] = None,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> WorkflowDurationMetrics:
        """
        Start tracking a workflow's duration.

        Args:
            workflow_id: Unique identifier for the workflow
            source: Source of the workflow (CLI, SLACK, etc.)
            workflow_type: Type of workflow (FAST_REPLY, COMPLEX_WORKFLOW)
            instruction: The user instruction/prompt
            user_id: User identifier (for Slack workflows)
            channel_id: Channel identifier (for Slack workflows)

        Returns:
            WorkflowDurationMetrics: The created metrics object
        """
        start_time = time.time()

        metrics = WorkflowDurationMetrics(
            workflow_id=workflow_id,
            source=source,
            workflow_type=workflow_type,
            start_time=start_time,
            instruction=instruction,
            user_id=user_id,
            channel_id=channel_id,
        )

        self.active_workflows[workflow_id] = metrics

        if self.enable_console_logging:
            self.logger.info(
                f"Started tracking workflow '{workflow_id}' from {source.value}"
            )

        return metrics

    def complete_workflow_tracking(
        self,
        workflow_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        role: Optional[str] = None,
        confidence: Optional[float] = None,
        task_count: Optional[int] = None,
    ) -> Optional[WorkflowDurationMetrics]:
        """
        Complete tracking for a workflow and log the duration.

        Args:
            workflow_id: Unique identifier for the workflow
            success: Whether the workflow completed successfully
            error_message: Error message if workflow failed
            role: Role used for execution (for fast-reply workflows)
            confidence: Routing confidence (for fast-reply workflows)
            task_count: Number of tasks in the workflow

        Returns:
            WorkflowDurationMetrics: The completed metrics object, or None if not found
        """
        if workflow_id not in self.active_workflows:
            self.logger.warning(
                f"Attempted to complete tracking for unknown workflow '{workflow_id}'"
            )
            return None

        metrics = self.active_workflows.pop(workflow_id)
        metrics.complete(success=success, error_message=error_message)

        # Add additional metadata
        if role:
            metrics.role = role
        if confidence is not None:
            metrics.confidence = confidence
        if task_count is not None:
            metrics.task_count = task_count

        # Log the completed workflow
        self._log_workflow_completion(metrics)

        if self.enable_console_logging:
            status = "✅ completed" if success else "❌ failed"
            self.logger.info(
                f"Workflow '{workflow_id}' {status} in {metrics.duration_seconds:.2f}s "
                f"(source: {metrics.source.value}, type: {metrics.workflow_type.value})"
            )

        return metrics

    def update_workflow_type(self, workflow_id: str, workflow_type: WorkflowType):
        """Update the workflow type for an active workflow."""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id].workflow_type = workflow_type

    def get_active_workflow_count(self) -> int:
        """Get the number of currently active workflows being tracked."""
        return len(self.active_workflows)

    def get_active_workflows(self) -> List[WorkflowDurationMetrics]:
        """Get list of currently active workflows."""
        return list(self.active_workflows.values())

    def _log_workflow_completion(self, metrics: WorkflowDurationMetrics):
        """Log workflow completion to file."""
        try:
            # Write to JSONL file
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                json.dump(metrics.to_dict(), f, ensure_ascii=False)
                f.write("\n")

            # Rotate log file if needed
            self._rotate_log_file_if_needed()

        except Exception as e:
            self.logger.error(f"Failed to log workflow completion to file: {e}")

    def _rotate_log_file_if_needed(self):
        """Rotate log file if it exceeds the maximum size."""
        try:
            if os.path.exists(self.log_file_path):
                file_size = os.path.getsize(self.log_file_path)
                if file_size > self.max_log_file_size_bytes:
                    # Create backup with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = f"{self.log_file_path}.{timestamp}"
                    os.rename(self.log_file_path, backup_path)

                    # Create new empty log file
                    with open(self.log_file_path, "w", encoding="utf-8") as _:
                        pass  # Create empty file

                    self.logger.info(f"Rotated log file to {backup_path}")
        except Exception as e:
            self.logger.error(f"Failed to rotate log file: {e}")

    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent workflow metrics from the log file.

        Args:
            limit: Maximum number of recent entries to return

        Returns:
            List of workflow metrics dictionaries
        """
        metrics = []
        try:
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # Get the last 'limit' lines
                    recent_lines = lines[-limit:] if len(lines) > limit else lines

                    for line in recent_lines:
                        try:
                            metrics.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            self.logger.error(f"Failed to read recent metrics: {e}")

        # Return in reverse order (most recent first)
        return list(reversed(metrics))

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary for the last N hours.

        Args:
            hours: Number of hours to analyze

        Returns:
            Dictionary with performance statistics
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = self.get_recent_metrics(
            limit=1000
        )  # Get more data for analysis

        # Filter to the specified time window
        filtered_metrics = [
            m for m in recent_metrics if m.get("start_time", 0) >= cutoff_time
        ]

        if not filtered_metrics:
            return {"message": f"No workflow data found for the last {hours} hours"}

        # Calculate statistics
        durations = [
            m.get("duration_seconds", 0)
            for m in filtered_metrics
            if m.get("duration_seconds")
        ]
        successful = [m for m in filtered_metrics if m.get("success", True)]
        failed = [m for m in filtered_metrics if not m.get("success", True)]

        by_source = {}
        by_type = {}

        for m in filtered_metrics:
            source = m.get("source", "UNKNOWN")
            wf_type = m.get("workflow_type", "UNKNOWN")

            by_source[source] = by_source.get(source, 0) + 1
            by_type[wf_type] = by_type.get(wf_type, 0) + 1

        return {
            "time_window_hours": hours,
            "total_workflows": len(filtered_metrics),
            "successful_workflows": len(successful),
            "failed_workflows": len(failed),
            "success_rate": (
                len(successful) / len(filtered_metrics) if filtered_metrics else 0
            ),
            "average_duration_seconds": (
                sum(durations) / len(durations) if durations else 0
            ),
            "min_duration_seconds": min(durations) if durations else 0,
            "max_duration_seconds": max(durations) if durations else 0,
            "workflows_by_source": by_source,
            "workflows_by_type": by_type,
        }


# Global instance for easy access
_global_duration_logger: Optional[WorkflowDurationLogger] = None


def get_duration_logger() -> WorkflowDurationLogger:
    """Get the global workflow duration logger instance."""
    global _global_duration_logger
    if _global_duration_logger is None:
        _global_duration_logger = WorkflowDurationLogger()
    return _global_duration_logger


def initialize_duration_logger(
    log_file_path: str = "logs/workflow_durations.jsonl",
    enable_console_logging: bool = True,
    max_log_file_size_mb: int = 100,
) -> WorkflowDurationLogger:
    """Initialize the global workflow duration logger."""
    global _global_duration_logger
    _global_duration_logger = WorkflowDurationLogger(
        log_file_path=log_file_path,
        enable_console_logging=enable_console_logging,
        max_log_file_size_mb=max_log_file_size_mb,
    )
    return _global_duration_logger
