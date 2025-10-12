"""
Tests for the Intent System Foundation

Tests the core intent classes and validation logic to ensure
LLM-safe declarative event processing works correctly.

Following TDD principles - tests written first.
"""

import time
from unittest.mock import Mock

import pytest

from common.intents import (
    AuditIntent,
    ErrorIntent,
    Intent,
    NotificationIntent,
    WorkflowIntent,
    create_error_intent,
    create_notification_from_error,
    validate_intent_list,
)


class TestNotificationIntent:
    """Test NotificationIntent validation and creation."""

    def test_valid_notification_intent(self):
        """Test creating a valid notification intent."""
        intent = NotificationIntent(message="Test message", channel="test-channel")

        assert intent.validate() is True
        assert intent.message == "Test message"
        assert intent.channel == "test-channel"
        assert intent.priority == "medium"  # default
        assert intent.notification_type == "info"  # default
        assert intent.created_at is not None

    def test_notification_intent_with_all_fields(self):
        """Test notification intent with all fields specified."""
        intent = NotificationIntent(
            message="Important message",
            channel="alerts",
            user_id="user123",
            priority="high",
            notification_type="warning",
        )

        assert intent.validate() is True
        assert intent.user_id == "user123"
        assert intent.priority == "high"
        assert intent.notification_type == "warning"

    def test_invalid_notification_intent_empty_message(self):
        """Test notification intent with empty message."""
        intent = NotificationIntent(message="", channel="test-channel")

        assert intent.validate() is False

    def test_invalid_notification_intent_whitespace_message(self):
        """Test notification intent with whitespace-only message."""
        intent = NotificationIntent(message="   ", channel="test-channel")

        assert intent.validate() is False

    def test_invalid_notification_intent_bad_priority(self):
        """Test notification intent with invalid priority."""
        intent = NotificationIntent(
            message="Test message", channel="test-channel", priority="invalid"
        )

        assert intent.validate() is False

    def test_invalid_notification_intent_bad_type(self):
        """Test notification intent with invalid notification type."""
        intent = NotificationIntent(
            message="Test message", channel="test-channel", notification_type="invalid"
        )

        assert intent.validate() is False


class TestAuditIntent:
    """Test AuditIntent validation and creation."""

    def test_valid_audit_intent(self):
        """Test creating a valid audit intent."""
        intent = AuditIntent(
            action="user_login", details={"user_id": "123", "ip": "192.168.1.1"}
        )

        assert intent.validate() is True
        assert intent.action == "user_login"
        assert intent.details["user_id"] == "123"
        assert intent.severity == "info"  # default

    def test_audit_intent_with_all_fields(self):
        """Test audit intent with all fields specified."""
        intent = AuditIntent(
            action="security_violation",
            details={"reason": "failed_auth", "attempts": 5},
            user_id="user123",
            severity="warning",
        )

        assert intent.validate() is True
        assert intent.user_id == "user123"
        assert intent.severity == "warning"

    def test_invalid_audit_intent_empty_action(self):
        """Test audit intent with empty action."""
        intent = AuditIntent(action="", details={"test": "data"})

        assert intent.validate() is False

    def test_invalid_audit_intent_non_dict_details(self):
        """Test audit intent with non-dict details."""
        intent = AuditIntent(action="test_action", details="not a dict")

        assert intent.validate() is False

    def test_invalid_audit_intent_bad_severity(self):
        """Test audit intent with invalid severity."""
        intent = AuditIntent(
            action="test_action", details={"test": "data"}, severity="invalid"
        )

        assert intent.validate() is False


class TestWorkflowIntent:
    """Test WorkflowIntent validation and creation."""

    def test_valid_workflow_intent(self):
        """Test creating a valid workflow intent."""
        intent = WorkflowIntent(
            workflow_type="data_processing",
            parameters={"input_file": "data.csv", "output_format": "json"},
        )

        assert intent.validate() is True
        assert intent.workflow_type == "data_processing"
        assert intent.parameters["input_file"] == "data.csv"
        assert intent.priority == 1  # default

    def test_workflow_intent_with_all_fields(self):
        """Test workflow intent with all fields specified."""
        intent = WorkflowIntent(
            workflow_type="urgent_processing",
            parameters={"data": "test"},
            priority=5,
            context={"source": "api", "user": "admin"},
        )

        assert intent.validate() is True
        assert intent.priority == 5
        assert intent.context["source"] == "api"

    def test_invalid_workflow_intent_empty_type(self):
        """Test workflow intent with empty workflow type."""
        intent = WorkflowIntent(workflow_type="", parameters={"test": "data"})

        assert intent.validate() is False

    def test_invalid_workflow_intent_non_dict_parameters(self):
        """Test workflow intent with non-dict parameters."""
        intent = WorkflowIntent(workflow_type="test_workflow", parameters="not a dict")

        assert intent.validate() is False

    def test_invalid_workflow_intent_negative_priority(self):
        """Test workflow intent with negative priority."""
        intent = WorkflowIntent(
            workflow_type="test_workflow", parameters={"test": "data"}, priority=-1
        )

        assert intent.validate() is False


class TestErrorIntent:
    """Test ErrorIntent validation and creation."""

    def test_valid_error_intent(self):
        """Test creating a valid error intent."""
        intent = ErrorIntent(
            error_type="ValueError",
            error_message="Invalid input provided",
            error_details={"input": "bad_value", "expected": "number"},
        )

        assert intent.validate() is True
        assert intent.error_type == "ValueError"
        assert intent.error_message == "Invalid input provided"
        assert intent.recoverable is True  # default

    def test_error_intent_with_all_fields(self):
        """Test error intent with all fields specified."""
        intent = ErrorIntent(
            error_type="SystemError",
            error_message="Critical system failure",
            error_details={"component": "database", "code": 500},
            recoverable=False,
            user_id="admin",
        )

        assert intent.validate() is True
        assert intent.recoverable is False
        assert intent.user_id == "admin"

    def test_invalid_error_intent_empty_type(self):
        """Test error intent with empty error type."""
        intent = ErrorIntent(
            error_type="", error_message="Some error", error_details={"test": "data"}
        )

        assert intent.validate() is False

    def test_invalid_error_intent_empty_message(self):
        """Test error intent with empty error message."""
        intent = ErrorIntent(
            error_type="TestError", error_message="", error_details={"test": "data"}
        )

        assert intent.validate() is False


class TestUtilityFunctions:
    """Test utility functions for intent handling."""

    def test_validate_intent_list_valid(self):
        """Test validating a list of valid intents."""
        intents = [
            NotificationIntent(message="Test", channel="test"),
            AuditIntent(action="test", details={"data": "test"}),
        ]

        assert validate_intent_list(intents) is True

    def test_validate_intent_list_invalid(self):
        """Test validating a list with invalid intents."""
        intents = [
            NotificationIntent(message="Test", channel="test"),
            NotificationIntent(message="", channel="test"),  # Invalid
        ]

        assert validate_intent_list(intents) is False

    def test_validate_intent_list_not_list(self):
        """Test validating non-list input."""
        assert validate_intent_list("not a list") is False
        assert validate_intent_list(None) is False

    def test_create_error_intent_from_exception(self):
        """Test creating error intent from exception."""
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            intent = create_error_intent(e, {"context": "test"})

            assert intent.validate() is True
            assert intent.error_type == "ValueError"
            assert intent.error_message == "Test error message"
            assert intent.error_details["context"]["context"] == "test"
            assert intent.recoverable is True

    def test_create_error_intent_system_exit(self):
        """Test creating error intent from SystemExit (non-recoverable)."""
        try:
            raise SystemExit("System shutdown")
        except SystemExit as e:
            intent = create_error_intent(e)

            assert intent.validate() is True
            assert intent.error_type == "SystemExit"
            assert intent.recoverable is False

    def test_create_notification_from_error(self):
        """Test creating notification intent from exception."""
        try:
            raise RuntimeError("Something went wrong")
        except RuntimeError as e:
            intent = create_notification_from_error(e, "alerts")

            assert intent.validate() is True
            assert "RuntimeError" in intent.message
            assert "Something went wrong" in intent.message
            assert intent.channel == "alerts"
            assert intent.priority == "high"
            assert intent.notification_type == "error"


class TestIntentSerialization:
    """Test intent serialization functionality."""

    def test_intent_to_dict(self):
        """Test converting intent to dictionary."""
        intent = NotificationIntent(message="Test message", channel="test-channel")

        result = intent.to_dict()

        assert result["type"] == "NotificationIntent"
        assert result["data"]["message"] == "Test message"
        assert result["data"]["channel"] == "test-channel"
        assert "created_at" in result
        assert isinstance(result["created_at"], float)


if __name__ == "__main__":
    pytest.main([__file__])
