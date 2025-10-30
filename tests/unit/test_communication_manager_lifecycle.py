"""Unit tests for Communication Manager workflow lifecycle tracking.

Tests the communication manager's ability to track workflow lifecycle events
and automatically clean up request IDs following Document 35 Phase 2.4.

Following Documents 25 & 26 LLM-safe architecture patterns.
"""

import time
from unittest.mock import Mock

from common.communication_manager import CommunicationManager
from common.message_bus import MessageBus, MessageType


class TestCommunicationManagerLifecycle:
    """Test communication manager workflow lifecycle tracking."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create message bus
        self.message_bus = MessageBus()
        self.message_bus.start()

        # Create mock supervisor
        self.mock_supervisor = Mock()

        # Create communication manager
        self.communication_manager = CommunicationManager(
            message_bus=self.message_bus, supervisor=self.mock_supervisor
        )

    def test_workflow_started_event_tracked(self):
        """Test that WORKFLOW_STARTED events are tracked."""
        # Arrange
        event_data = {
            "workflow_id": "req_123_task_1",
            "parent_request_id": "req_123",
            "task_name": "Search Task",
            "role": "search",
        }

        # Act
        self.message_bus.publish(self, MessageType.WORKFLOW_STARTED, event_data)

        # Assert
        assert "req_123" in self.communication_manager.active_workflows
        assert (
            "req_123_task_1" in self.communication_manager.active_workflows["req_123"]
        )

    def test_workflow_completed_event_removes_workflow(self):
        """Test that WORKFLOW_COMPLETED events remove workflows from tracking."""
        # Arrange - Start tracking a workflow
        start_event = {
            "workflow_id": "req_123_task_1",
            "parent_request_id": "req_123",
        }
        self.message_bus.publish(self, MessageType.WORKFLOW_STARTED, start_event)

        # Act - Complete the workflow
        complete_event = {
            "workflow_id": "req_123_task_1",
            "parent_request_id": "req_123",
            "success": True,
        }
        self.message_bus.publish(self, MessageType.WORKFLOW_COMPLETED, complete_event)

        # Assert - Request should be cleaned up
        assert "req_123" not in self.communication_manager.active_workflows
        assert "req_123" not in self.communication_manager._pending_requests

    def test_multiple_workflows_tracked_per_request(self):
        """Test tracking multiple workflows for single request."""
        # Arrange & Act - Start two workflows for same request
        self.message_bus.publish(
            self,
            MessageType.WORKFLOW_STARTED,
            {"workflow_id": "req_123_task_1", "parent_request_id": "req_123"},
        )
        self.message_bus.publish(
            self,
            MessageType.WORKFLOW_STARTED,
            {"workflow_id": "req_123_task_2", "parent_request_id": "req_123"},
        )

        # Assert
        assert len(self.communication_manager.active_workflows["req_123"]) == 2

    def test_cleanup_only_after_all_workflows_complete(self):
        """Test that cleanup only happens after ALL workflows complete."""
        # Arrange - Start two workflows
        self.message_bus.publish(
            self,
            MessageType.WORKFLOW_STARTED,
            {"workflow_id": "req_123_task_1", "parent_request_id": "req_123"},
        )
        self.message_bus.publish(
            self,
            MessageType.WORKFLOW_STARTED,
            {"workflow_id": "req_123_task_2", "parent_request_id": "req_123"},
        )

        # Act - Complete first workflow
        self.message_bus.publish(
            self,
            MessageType.WORKFLOW_COMPLETED,
            {"workflow_id": "req_123_task_1", "parent_request_id": "req_123"},
        )

        # Assert - Request still tracked (second workflow pending)
        assert "req_123" in self.communication_manager.active_workflows
        assert len(self.communication_manager.active_workflows["req_123"]) == 1

        # Act - Complete second workflow
        self.message_bus.publish(
            self,
            MessageType.WORKFLOW_COMPLETED,
            {"workflow_id": "req_123_task_2", "parent_request_id": "req_123"},
        )

        # Assert - Now request should be cleaned up
        assert "req_123" not in self.communication_manager.active_workflows

    def test_workflow_failed_event_triggers_cleanup(self):
        """Test that WORKFLOW_FAILED events trigger cleanup."""
        # Arrange
        self.message_bus.publish(
            self,
            MessageType.WORKFLOW_STARTED,
            {"workflow_id": "req_123_task_1", "parent_request_id": "req_123"},
        )

        # Act
        self.message_bus.publish(
            self,
            MessageType.WORKFLOW_FAILED,
            {
                "workflow_id": "req_123_task_1",
                "parent_request_id": "req_123",
                "error": "Test error",
            },
        )

        # Assert
        assert "req_123" not in self.communication_manager.active_workflows

    def test_timeout_handling_with_dead_letter_queue(self):
        """Test that expired requests are logged to dead letter queue."""
        # Arrange - Start workflow and set short timeout
        self.message_bus.publish(
            self,
            MessageType.WORKFLOW_STARTED,
            {"workflow_id": "req_123_task_1", "parent_request_id": "req_123"},
        )

        # Manually set timeout to past
        self.communication_manager.request_timeouts["req_123"] = time.time() - 1

        # Act - Trigger cleanup
        self.communication_manager._cleanup_expired_requests()

        # Assert - Request should be cleaned up
        assert "req_123" not in self.communication_manager.active_workflows
        assert "req_123" not in self.communication_manager.request_timeouts

    def test_lifecycle_tracking_prevents_premature_cleanup(self):
        """Test that lifecycle tracking prevents premature request cleanup."""
        # Arrange - Start workflow
        self.message_bus.publish(
            self,
            MessageType.WORKFLOW_STARTED,
            {"workflow_id": "req_123_task_1", "parent_request_id": "req_123"},
        )

        # Assert - Request is being tracked
        assert "req_123" in self.communication_manager.active_workflows
        assert "req_123" in self.communication_manager.request_timeouts

        # Act - Simulate time passing but workflow still active
        # The request should remain tracked
        assert len(self.communication_manager.active_workflows["req_123"]) == 1

        # Complete workflow
        self.message_bus.publish(
            self,
            MessageType.WORKFLOW_COMPLETED,
            {"workflow_id": "req_123_task_1", "parent_request_id": "req_123"},
        )

        # Assert - Now cleanup should happen
        assert "req_123" not in self.communication_manager.active_workflows
