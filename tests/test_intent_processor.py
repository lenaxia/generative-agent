"""
Tests for the Intent Processor

Tests the core intent processing logic to ensure LLM-safe declarative
event processing works correctly with proper error handling.

Following TDD principles - tests written first.
"""

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from common.intent_processor import IntentProcessor
from common.intents import (
    AuditIntent,
    ErrorIntent,
    Intent,
    NotificationIntent,
    WorkflowIntent,
)


class TestIntentProcessor:
    """Test IntentProcessor core functionality."""

    @pytest.fixture
    def mock_communication_manager(self):
        """Create mock communication manager."""
        mock = AsyncMock()
        mock.send_notification = AsyncMock()
        return mock

    @pytest.fixture
    def mock_workflow_engine(self):
        """Create mock workflow engine."""
        mock = AsyncMock()
        mock.start_workflow = AsyncMock(return_value="workflow_123")
        return mock

    @pytest.fixture
    def intent_processor(self, mock_communication_manager, mock_workflow_engine):
        """Create intent processor with mocked dependencies."""
        return IntentProcessor(
            communication_manager=mock_communication_manager,
            workflow_engine=mock_workflow_engine,
        )

    @pytest.mark.asyncio
    async def test_process_notification_intent(
        self, intent_processor, mock_communication_manager
    ):
        """Test processing notification intent."""
        intent = NotificationIntent(
            message="Test notification",
            channel="test-channel",
            user_id="user123",
            priority="high",
        )

        result = await intent_processor.process_intents([intent])

        assert result["processed"] == 1
        assert result["failed"] == 0
        assert len(result["errors"]) == 0

        mock_communication_manager.send_notification.assert_called_once_with(
            message="Test notification", channel="test-channel", user_id="user123"
        )

    @pytest.mark.asyncio
    async def test_process_audit_intent(self, intent_processor):
        """Test processing audit intent."""
        intent = AuditIntent(
            action="user_login",
            details={"user_id": "123", "ip": "192.168.1.1"},
            user_id="user123",
            severity="info",
        )

        result = await intent_processor.process_intents([intent])

        assert result["processed"] == 1
        assert result["failed"] == 0
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_process_workflow_intent(
        self, intent_processor, mock_workflow_engine
    ):
        """Test processing workflow intent."""
        intent = WorkflowIntent(
            workflow_type="data_processing",
            parameters={"input": "test.csv"},
            priority=5,
        )

        result = await intent_processor.process_intents([intent])

        assert result["processed"] == 1
        assert result["failed"] == 0
        assert len(result["errors"]) == 0

        mock_workflow_engine.start_workflow.assert_called_once_with(
            request="Execute data_processing", parameters={"input": "test.csv"}
        )

    @pytest.mark.asyncio
    async def test_process_error_intent(self, intent_processor):
        """Test processing error intent."""
        intent = ErrorIntent(
            error_type="ValueError",
            error_message="Invalid input",
            error_details={"input": "bad_value"},
            recoverable=True,
        )

        result = await intent_processor.process_intents([intent])

        assert result["processed"] == 1
        assert result["failed"] == 0
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_process_multiple_intents(
        self, intent_processor, mock_communication_manager, mock_workflow_engine
    ):
        """Test processing multiple intents in one call."""
        intents = [
            NotificationIntent(message="Test 1", channel="test"),
            AuditIntent(action="test_action", details={"key": "value"}),
            WorkflowIntent(
                workflow_type="test_workflow", parameters={"param": "value"}
            ),
        ]

        result = await intent_processor.process_intents(intents)

        assert result["processed"] == 3
        assert result["failed"] == 0
        assert len(result["errors"]) == 0

        # Verify all intents were processed
        mock_communication_manager.send_notification.assert_called_once()
        mock_workflow_engine.start_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_invalid_intent(self, intent_processor):
        """Test processing invalid intent."""
        # Create invalid notification intent
        invalid_intent = NotificationIntent(message="", channel="test")  # Empty message
        valid_intent = NotificationIntent(message="Valid", channel="test")

        result = await intent_processor.process_intents([invalid_intent, valid_intent])

        assert result["processed"] == 1  # Only valid intent processed
        assert result["failed"] == 1  # Invalid intent failed
        assert len(result["errors"]) == 1
        assert "Invalid intent" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_process_intent_with_missing_communication_manager(self):
        """Test processing notification intent without communication manager."""
        processor = IntentProcessor(communication_manager=None, workflow_engine=None)
        intent = NotificationIntent(message="Test", channel="test")

        result = await processor.process_intents([intent])

        # Should still process successfully but log error
        assert result["processed"] == 1
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_process_intent_with_missing_workflow_engine(self):
        """Test processing workflow intent without workflow engine."""
        processor = IntentProcessor(communication_manager=None, workflow_engine=None)
        intent = WorkflowIntent(workflow_type="test", parameters={})

        result = await processor.process_intents([intent])

        # Should still process successfully but log error
        assert result["processed"] == 1
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_process_intent_with_exception(
        self, intent_processor, mock_communication_manager
    ):
        """Test processing intent when communication manager raises exception."""
        mock_communication_manager.send_notification.side_effect = Exception(
            "Network error"
        )

        intent = NotificationIntent(message="Test", channel="test")

        result = await intent_processor.process_intents([intent])

        assert result["processed"] == 0
        assert result["failed"] == 1
        assert len(result["errors"]) == 1
        assert "Network error" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_register_role_intent_handler(self, intent_processor):
        """Test registering custom role intent handler."""
        # Create a custom intent type for testing
        from dataclasses import dataclass

        @dataclass
        class CustomIntent(Intent):
            action: str

            def validate(self) -> bool:
                return bool(self.action)

        # Create mock handler
        mock_handler = AsyncMock()

        # Register the handler
        intent_processor.register_role_intent_handler(
            CustomIntent, mock_handler, "test_role"
        )

        # Create and process custom intent
        custom_intent = CustomIntent(action="test_action")
        result = await intent_processor.process_intents([custom_intent])

        assert result["processed"] == 1
        assert result["failed"] == 0
        mock_handler.assert_called_once_with(custom_intent)

    @pytest.mark.asyncio
    async def test_process_unknown_intent_type(self, intent_processor):
        """Test processing unknown intent type."""
        # Create a custom intent that's not registered
        from dataclasses import dataclass

        @dataclass
        class UnknownIntent(Intent):
            action: str

            def validate(self) -> bool:
                return True

        unknown_intent = UnknownIntent(action="test")
        result = await intent_processor.process_intents([unknown_intent])

        # Should still count as processed (logged as warning)
        assert result["processed"] == 1
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_intent_processor_metrics(self, intent_processor):
        """Test intent processor tracks metrics correctly."""
        intents = [
            NotificationIntent(message="Test 1", channel="test"),
            NotificationIntent(message="Test 2", channel="test"),
            AuditIntent(action="test", details={}),
        ]

        await intent_processor.process_intents(intents)

        # Check that processor tracks total processed count
        assert intent_processor.get_processed_count() == 3

    def test_intent_processor_initialization(self):
        """Test intent processor initializes correctly."""
        comm_manager = MagicMock()
        workflow_engine = MagicMock()

        processor = IntentProcessor(
            communication_manager=comm_manager, workflow_engine=workflow_engine
        )

        assert processor.communication_manager == comm_manager
        assert processor.workflow_engine == workflow_engine
        assert processor.get_processed_count() == 0

    def test_intent_processor_initialization_without_dependencies(self):
        """Test intent processor initializes without dependencies."""
        processor = IntentProcessor()

        assert processor.communication_manager is None
        assert processor.workflow_engine is None
        assert processor.get_processed_count() == 0


class TestIntentProcessorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_process_empty_intent_list(self):
        """Test processing empty intent list."""
        processor = IntentProcessor()
        result = await processor.process_intents([])

        assert result["processed"] == 0
        assert result["failed"] == 0
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_process_none_intent_list(self):
        """Test processing None as intent list."""
        processor = IntentProcessor()

        # Should handle gracefully
        with pytest.raises(TypeError):
            await processor.process_intents(None)

    @pytest.mark.asyncio
    async def test_concurrent_intent_processing(self):
        """Test concurrent intent processing doesn't cause issues."""
        mock_comm_manager = AsyncMock()
        processor = IntentProcessor(communication_manager=mock_comm_manager)

        # Create multiple intent lists to process concurrently
        intent_lists = [
            [NotificationIntent(message=f"Test {i}", channel="test") for i in range(5)]
            for _ in range(3)
        ]

        # Process all lists concurrently
        tasks = [processor.process_intents(intents) for intents in intent_lists]
        results = await asyncio.gather(*tasks)

        # Verify all were processed successfully
        for result in results:
            assert result["processed"] == 5
            assert result["failed"] == 0

        # Verify total processed count
        assert processor.get_processed_count() == 15


if __name__ == "__main__":
    pytest.main([__file__])
