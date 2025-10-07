"""
Tests for Workflow Duration Logger

Comprehensive test suite for workflow duration tracking and logging functionality.
"""

import json
import os
import tempfile
import time

import pytest

from supervisor.workflow_duration_logger import (
    WorkflowDurationLogger,
    WorkflowDurationMetrics,
    WorkflowSource,
    WorkflowType,
    get_duration_logger,
    initialize_duration_logger,
)


class TestWorkflowDurationMetrics:
    """Test WorkflowDurationMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating workflow duration metrics."""
        start_time = time.time()
        metrics = WorkflowDurationMetrics(
            workflow_id="test_workflow_123",
            source=WorkflowSource.CLI,
            workflow_type=WorkflowType.FAST_REPLY,
            start_time=start_time,
            instruction="Test workflow instruction",
        )

        assert metrics.workflow_id == "test_workflow_123"
        assert metrics.source == WorkflowSource.CLI
        assert metrics.workflow_type == WorkflowType.FAST_REPLY
        assert metrics.start_time == start_time
        assert metrics.instruction == "Test workflow instruction"
        assert metrics.end_time is None
        assert metrics.duration_seconds is None
        assert metrics.success is True
        assert metrics.timestamp is not None

    def test_metrics_completion(self):
        """Test completing workflow metrics."""
        start_time = time.time()
        metrics = WorkflowDurationMetrics(
            workflow_id="test_workflow_123",
            source=WorkflowSource.SLACK,
            workflow_type=WorkflowType.COMPLEX_WORKFLOW,
            start_time=start_time,
        )

        # Simulate some execution time
        time.sleep(0.1)
        end_time = time.time()

        metrics.complete(end_time=end_time, success=True)

        assert metrics.end_time == end_time
        assert metrics.duration_seconds is not None
        assert metrics.duration_seconds > 0
        assert metrics.success is True
        assert metrics.error_message is None

    def test_metrics_failure(self):
        """Test failing workflow metrics."""
        start_time = time.time()
        metrics = WorkflowDurationMetrics(
            workflow_id="test_workflow_123",
            source=WorkflowSource.CLI,
            workflow_type=WorkflowType.COMPLEX_WORKFLOW,
            start_time=start_time,
        )

        time.sleep(0.1)
        end_time = time.time()
        error_msg = "Test error message"

        metrics.complete(end_time=end_time, success=False, error_message=error_msg)

        assert metrics.end_time == end_time
        assert metrics.duration_seconds > 0
        assert metrics.success is False
        assert metrics.error_message == error_msg

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        start_time = time.time()
        metrics = WorkflowDurationMetrics(
            workflow_id="test_workflow_123",
            source=WorkflowSource.SLACK,
            workflow_type=WorkflowType.FAST_REPLY,
            start_time=start_time,
            instruction="Test instruction",
            user_id="user123",
            channel_id="channel456",
        )

        metrics.complete(success=True)

        data = metrics.to_dict()

        assert isinstance(data, dict)
        assert data["workflow_id"] == "test_workflow_123"
        assert data["source"] == WorkflowSource.SLACK.value
        assert data["workflow_type"] == WorkflowType.FAST_REPLY.value
        assert data["start_time"] == start_time
        assert data["instruction"] == "Test instruction"
        assert data["user_id"] == "user123"
        assert data["channel_id"] == "channel456"
        assert data["success"] is True
        assert "duration_seconds" in data
        assert "timestamp" in data


class TestWorkflowDurationLogger:
    """Test WorkflowDurationLogger class."""

    def setup_method(self):
        """Setup test environment."""
        # Create temporary log file
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_workflow_durations.jsonl")

        # Create logger instance
        self.logger = WorkflowDurationLogger(
            log_file_path=self.log_file,
            enable_console_logging=False,  # Disable for tests
            max_log_file_size_mb=1,
        )

    def teardown_method(self):
        """Cleanup test environment."""
        # Clean up temp files
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_logger_initialization(self):
        """Test logger initialization."""
        assert self.logger.log_file_path == self.log_file
        assert self.logger.enable_console_logging is False
        assert self.logger.max_log_file_size_bytes == 1024 * 1024  # 1MB
        assert len(self.logger.active_workflows) == 0

    def test_start_workflow_tracking(self):
        """Test starting workflow tracking."""
        workflow_id = "test_workflow_123"
        instruction = "Test workflow instruction"

        metrics = self.logger.start_workflow_tracking(
            workflow_id=workflow_id,
            source=WorkflowSource.CLI,
            workflow_type=WorkflowType.COMPLEX_WORKFLOW,
            instruction=instruction,
        )

        assert metrics.workflow_id == workflow_id
        assert metrics.source == WorkflowSource.CLI
        assert metrics.workflow_type == WorkflowType.COMPLEX_WORKFLOW
        assert metrics.instruction == instruction
        assert workflow_id in self.logger.active_workflows
        assert self.logger.get_active_workflow_count() == 1

    def test_complete_workflow_tracking(self):
        """Test completing workflow tracking."""
        workflow_id = "test_workflow_123"

        # Start tracking
        self.logger.start_workflow_tracking(
            workflow_id=workflow_id,
            source=WorkflowSource.SLACK,
            instruction="Test instruction",
        )

        # Simulate some execution time
        time.sleep(0.1)

        # Complete tracking
        completed_metrics = self.logger.complete_workflow_tracking(
            workflow_id=workflow_id,
            success=True,
            role="weather",
            confidence=0.95,
            task_count=3,
        )

        assert completed_metrics is not None
        assert completed_metrics.workflow_id == workflow_id
        assert completed_metrics.success is True
        assert completed_metrics.role == "weather"
        assert completed_metrics.confidence == 0.95
        assert completed_metrics.task_count == 3
        assert completed_metrics.duration_seconds > 0
        assert workflow_id not in self.logger.active_workflows
        assert self.logger.get_active_workflow_count() == 0

        # Check that log file was created and contains the entry
        assert os.path.exists(self.log_file)
        with open(self.log_file, "r") as f:
            log_entry = json.loads(f.read().strip())
            assert log_entry["workflow_id"] == workflow_id
            assert log_entry["success"] is True
            assert log_entry["role"] == "weather"

    def test_complete_unknown_workflow(self):
        """Test completing tracking for unknown workflow."""
        result = self.logger.complete_workflow_tracking(
            workflow_id="unknown_workflow", success=True
        )

        assert result is None

    def test_update_workflow_type(self):
        """Test updating workflow type."""
        workflow_id = "test_workflow_123"

        # Start tracking with unknown type
        self.logger.start_workflow_tracking(
            workflow_id=workflow_id,
            source=WorkflowSource.CLI,
            workflow_type=WorkflowType.UNKNOWN,
        )

        # Update type
        self.logger.update_workflow_type(workflow_id, WorkflowType.FAST_REPLY)

        # Check that type was updated
        active_workflow = self.logger.active_workflows[workflow_id]
        assert active_workflow.workflow_type == WorkflowType.FAST_REPLY

    def test_workflow_failure_tracking(self):
        """Test tracking failed workflows."""
        workflow_id = "failed_workflow_123"
        error_message = "Test error occurred"

        # Start and complete with failure
        self.logger.start_workflow_tracking(
            workflow_id=workflow_id,
            source=WorkflowSource.CLI,
            instruction="Test instruction",
        )

        time.sleep(0.1)

        completed_metrics = self.logger.complete_workflow_tracking(
            workflow_id=workflow_id, success=False, error_message=error_message
        )

        assert completed_metrics.success is False
        assert completed_metrics.error_message == error_message

        # Check log file
        with open(self.log_file, "r") as f:
            log_entry = json.loads(f.read().strip())
            assert log_entry["success"] is False
            assert log_entry["error_message"] == error_message

    def test_multiple_workflows(self):
        """Test tracking multiple concurrent workflows."""
        workflow_ids = ["workflow_1", "workflow_2", "workflow_3"]

        # Start multiple workflows
        for wf_id in workflow_ids:
            self.logger.start_workflow_tracking(
                workflow_id=wf_id,
                source=WorkflowSource.SLACK,
                instruction=f"Instruction for {wf_id}",
            )

        assert self.logger.get_active_workflow_count() == 3

        # Complete them in different order
        self.logger.complete_workflow_tracking(workflow_ids[1], success=True)
        assert self.logger.get_active_workflow_count() == 2

        self.logger.complete_workflow_tracking(workflow_ids[0], success=True)
        assert self.logger.get_active_workflow_count() == 1

        self.logger.complete_workflow_tracking(
            workflow_ids[2], success=False, error_message="Test error"
        )
        assert self.logger.get_active_workflow_count() == 0

        # Check that all entries were logged
        with open(self.log_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 3

    def test_get_recent_metrics(self):
        """Test getting recent metrics from log file."""
        # Create some test entries
        workflow_ids = ["workflow_1", "workflow_2", "workflow_3"]

        for wf_id in workflow_ids:
            self.logger.start_workflow_tracking(
                workflow_id=wf_id,
                source=WorkflowSource.CLI,
                instruction=f"Instruction for {wf_id}",
            )
            time.sleep(0.01)  # Small delay to ensure different timestamps
            self.logger.complete_workflow_tracking(wf_id, success=True)

        # Get recent metrics
        recent_metrics = self.logger.get_recent_metrics(limit=2)

        assert len(recent_metrics) == 2
        # Should be in reverse order (most recent first)
        assert recent_metrics[0]["workflow_id"] == "workflow_3"
        assert recent_metrics[1]["workflow_id"] == "workflow_2"

    def test_get_performance_summary(self):
        """Test getting performance summary."""
        # Create test workflows with different outcomes
        successful_workflows = ["success_1", "success_2"]
        failed_workflows = ["failed_1"]

        # Create successful workflows
        for wf_id in successful_workflows:
            self.logger.start_workflow_tracking(
                workflow_id=wf_id,
                source=WorkflowSource.CLI,
                workflow_type=WorkflowType.FAST_REPLY,
                instruction=f"Instruction for {wf_id}",
            )
            time.sleep(0.01)
            self.logger.complete_workflow_tracking(wf_id, success=True, role="weather")

        # Create failed workflow
        for wf_id in failed_workflows:
            self.logger.start_workflow_tracking(
                workflow_id=wf_id,
                source=WorkflowSource.SLACK,
                workflow_type=WorkflowType.COMPLEX_WORKFLOW,
                instruction=f"Instruction for {wf_id}",
            )
            time.sleep(0.01)
            self.logger.complete_workflow_tracking(
                wf_id, success=False, error_message="Test error"
            )

        # Get performance summary
        summary = self.logger.get_performance_summary(hours=24)

        assert summary["total_workflows"] == 3
        assert summary["successful_workflows"] == 2
        assert summary["failed_workflows"] == 1
        assert summary["success_rate"] == 2 / 3
        assert summary["average_duration_seconds"] > 0
        assert summary["workflows_by_source"]["CLI"] == 2
        assert summary["workflows_by_source"]["SLACK"] == 1
        assert summary["workflows_by_type"]["FAST_REPLY"] == 2
        assert summary["workflows_by_type"]["COMPLEX_WORKFLOW"] == 1

    def test_log_file_rotation(self):
        """Test log file rotation when size limit is exceeded."""
        # Create logger with very small size limit
        small_logger = WorkflowDurationLogger(
            log_file_path=self.log_file,
            enable_console_logging=False,
            max_log_file_size_mb=0.001,  # Very small limit
        )

        # Create many workflows to exceed size limit
        for i in range(10):
            wf_id = f"workflow_{i}"
            small_logger.start_workflow_tracking(
                workflow_id=wf_id,
                source=WorkflowSource.CLI,
                instruction=f"Long instruction for workflow {i} "
                * 100,  # Make it large
            )
            small_logger.complete_workflow_tracking(wf_id, success=True)

        # Check that original file still exists (rotation creates backup)
        assert os.path.exists(self.log_file)


class TestGlobalDurationLogger:
    """Test global duration logger functions."""

    def test_get_duration_logger(self):
        """Test getting global duration logger."""
        logger1 = get_duration_logger()
        logger2 = get_duration_logger()

        # Should return the same instance
        assert logger1 is logger2

    def test_initialize_duration_logger(self):
        """Test initializing global duration logger."""
        temp_dir = tempfile.mkdtemp()
        log_file = os.path.join(temp_dir, "test_durations.jsonl")

        try:
            logger = initialize_duration_logger(
                log_file_path=log_file,
                enable_console_logging=False,
                max_log_file_size_mb=5,
            )

            assert logger.log_file_path == log_file
            assert logger.enable_console_logging is False
            assert logger.max_log_file_size_bytes == 5 * 1024 * 1024

            # Should be the same as get_duration_logger
            assert get_duration_logger() is logger

        finally:
            # Cleanup
            if os.path.exists(log_file):
                os.remove(log_file)
            os.rmdir(temp_dir)


class TestWorkflowDurationIntegration:
    """Integration tests for workflow duration logging."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "integration_test_durations.jsonl")

        # Initialize global logger for integration tests
        self.logger = initialize_duration_logger(
            log_file_path=self.log_file, enable_console_logging=False
        )

    def teardown_method(self):
        """Cleanup test environment."""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        os.rmdir(self.temp_dir)

    def test_cli_workflow_simulation(self):
        """Test simulating a CLI workflow execution."""
        workflow_id = "cli_test_workflow"
        instruction = "Get weather for Seattle"

        # Simulate CLI workflow
        metrics = self.logger.start_workflow_tracking(
            workflow_id=workflow_id, source=WorkflowSource.CLI, instruction=instruction
        )

        # Simulate execution time
        time.sleep(0.1)

        # Determine workflow type (simulate fast-reply detection)
        if workflow_id.startswith("fr_"):
            workflow_type = WorkflowType.FAST_REPLY
        else:
            workflow_type = WorkflowType.COMPLEX_WORKFLOW

        self.logger.update_workflow_type(workflow_id, workflow_type)

        # Complete with success
        completed_metrics = self.logger.complete_workflow_tracking(
            workflow_id=workflow_id,
            success=True,
            role="weather",
            confidence=0.92,
            task_count=2,
        )

        assert completed_metrics.source == WorkflowSource.CLI
        assert completed_metrics.workflow_type == WorkflowType.COMPLEX_WORKFLOW
        assert completed_metrics.success is True
        assert completed_metrics.duration_seconds > 0

        # Verify log entry
        recent_metrics = self.logger.get_recent_metrics(limit=1)
        assert len(recent_metrics) == 1
        assert recent_metrics[0]["workflow_id"] == workflow_id
        assert recent_metrics[0]["source"] == "CLI"

    def test_slack_workflow_simulation(self):
        """Test simulating a Slack workflow execution."""
        workflow_id = "fr_slack_test_workflow"  # Fast-reply workflow
        instruction = "What's the weather like?"
        user_id = "U123456789"
        channel_id = "C987654321"

        # Simulate Slack workflow
        metrics = self.logger.start_workflow_tracking(
            workflow_id=workflow_id,
            source=WorkflowSource.SLACK,
            instruction=instruction,
            user_id=user_id,
            channel_id=channel_id,
        )

        # Simulate fast execution
        time.sleep(0.05)

        # Update workflow type based on ID
        workflow_type = (
            WorkflowType.FAST_REPLY
            if workflow_id.startswith("fr_")
            else WorkflowType.COMPLEX_WORKFLOW
        )
        self.logger.update_workflow_type(workflow_id, workflow_type)

        # Complete with success
        completed_metrics = self.logger.complete_workflow_tracking(
            workflow_id=workflow_id, success=True, role="weather", confidence=0.88
        )

        assert completed_metrics.source == WorkflowSource.SLACK
        assert completed_metrics.workflow_type == WorkflowType.FAST_REPLY
        assert completed_metrics.user_id == user_id
        assert completed_metrics.channel_id == channel_id
        assert completed_metrics.success is True

        # Verify log entry
        recent_metrics = self.logger.get_recent_metrics(limit=1)
        assert len(recent_metrics) == 1
        assert recent_metrics[0]["user_id"] == user_id
        assert recent_metrics[0]["channel_id"] == channel_id

    def test_workflow_interruption_simulation(self):
        """Test simulating workflow interruption."""
        workflow_id = "interrupted_workflow"

        # Start workflow
        self.logger.start_workflow_tracking(
            workflow_id=workflow_id,
            source=WorkflowSource.CLI,
            instruction="Long running task",
        )

        # Simulate interruption
        time.sleep(0.05)

        completed_metrics = self.logger.complete_workflow_tracking(
            workflow_id=workflow_id,
            success=False,
            error_message="Workflow interrupted by user",
        )

        assert completed_metrics.success is False
        assert "interrupted" in completed_metrics.error_message.lower()

        # Verify in performance summary
        summary = self.logger.get_performance_summary(hours=1)
        assert summary["failed_workflows"] >= 1
        assert summary["success_rate"] < 1.0


if __name__ == "__main__":
    pytest.main([__file__])
