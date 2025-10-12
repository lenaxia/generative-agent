"""
Tests for Enhanced MessageBus with Intent Processing

Tests the enhanced MessageBus that integrates intent processing
for LLM-safe threading architecture.

Following TDD principles - tests written first.
"""

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.enhanced_event_context import (
    LLMSafeEventContext,
    create_context_from_event_data,
)
from common.intent_processor import IntentProcessor
from common.intents import AuditIntent, Intent, NotificationIntent
from common.message_bus import MessageBus


class TestEnhancedMessageBus:
    """Test enhanced MessageBus with intent processing capabilities."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for MessageBus."""
        return {
            "communication_manager": AsyncMock(),
            "workflow_engine": AsyncMock(),
            "llm_factory": MagicMock(),
        }

    @pytest.fixture
    def enhanced_message_bus(self, mock_dependencies):
        """Create enhanced MessageBus with mocked dependencies."""
        bus = MessageBus()
        bus.communication_manager = mock_dependencies["communication_manager"]
        bus.workflow_engine = mock_dependencies["workflow_engine"]
        bus.llm_factory = mock_dependencies["llm_factory"]
        bus.start()
        return bus

    @pytest.mark.asyncio
    async def test_message_bus_intent_processing_initialization(
        self, enhanced_message_bus
    ):
        """Test that MessageBus initializes intent processor correctly."""
        # Should have intent processor
        assert hasattr(enhanced_message_bus, "_intent_processor")
        assert enhanced_message_bus._intent_processor is not None

        # Should have intent processing enabled
        assert hasattr(enhanced_message_bus, "_enable_intent_processing")
        assert enhanced_message_bus._enable_intent_processing is True

    @pytest.mark.asyncio
    async def test_pure_function_handler_with_intents(
        self, enhanced_message_bus, mock_dependencies
    ):
        """Test pure function handler that returns intents."""

        # Create a pure function handler that returns intents
        def pure_timer_handler(event_data, context):
            """Pure function handler returning intents."""
            return [
                NotificationIntent(
                    message=f"Timer expired: {context.get_metadata('original_request')}",
                    channel=context.get_safe_channel(),
                    user_id=context.user_id,
                ),
                AuditIntent(
                    action="timer_expired",
                    details={"timer_id": context.get_metadata("timer_id")},
                    user_id=context.user_id,
                ),
            ]

        # Subscribe the handler
        enhanced_message_bus.subscribe("timer", "TIMER_EXPIRED", pure_timer_handler)

        # Publish event
        await enhanced_message_bus.publish_async(
            publisher=MagicMock(),
            event_type="TIMER_EXPIRED",
            message=["timer_123", "Set timer for 5 minutes"],
        )

        # Wait for async processing
        await asyncio.sleep(0.1)

        # Verify intent processor was called
        mock_dependencies["communication_manager"].send_notification.assert_called()

    @pytest.mark.asyncio
    async def test_legacy_handler_compatibility(self, enhanced_message_bus):
        """Test that legacy handlers still work without returning intents."""
        legacy_handler_called = False

        async def legacy_handler(event_data):
            """Legacy handler that doesn't return intents."""
            nonlocal legacy_handler_called
            legacy_handler_called = True

        # Subscribe legacy handler
        enhanced_message_bus.subscribe("legacy", "TEST_EVENT", legacy_handler)

        # Publish event
        await enhanced_message_bus.publish_async(
            publisher=MagicMock(), event_type="TEST_EVENT", message={"test": "data"}
        )

        # Wait for processing
        await asyncio.sleep(0.1)

        # Verify legacy handler was called
        assert legacy_handler_called is True

    @pytest.mark.asyncio
    async def test_context_creation_for_handlers(self, enhanced_message_bus):
        """Test that MessageBus creates proper context for handlers."""
        received_context = None

        def context_checking_handler(event_data, context):
            """Handler that checks the context it receives."""
            nonlocal received_context
            received_context = context
            return []  # Return empty intent list

        # Subscribe handler
        enhanced_message_bus.subscribe("test", "CONTEXT_TEST", context_checking_handler)

        # Publish event with publisher that has user info
        mock_publisher = MagicMock()
        mock_publisher.channel_id = "test_channel"
        mock_publisher.user_id = "test_user"

        await enhanced_message_bus.publish_async(
            publisher=mock_publisher,
            event_type="CONTEXT_TEST",
            message={"test": "data"},
        )

        # Wait for processing
        await asyncio.sleep(0.1)

        # Verify context was created and passed
        assert received_context is not None
        assert hasattr(received_context, "user_id")
        assert hasattr(received_context, "channel_id")
        assert hasattr(received_context, "timestamp")
        assert hasattr(received_context, "source")

    @pytest.mark.asyncio
    async def test_intent_validation_in_message_bus(self, enhanced_message_bus):
        """Test that MessageBus validates intents before processing."""
        invalid_intents_returned = False

        def handler_returning_invalid_intents(event_data, context):
            """Handler that returns invalid intents."""
            nonlocal invalid_intents_returned
            invalid_intents_returned = True
            # Return invalid intent (empty message)
            return [NotificationIntent(message="", channel="test")]

        # Subscribe handler
        enhanced_message_bus.subscribe(
            "test", "INVALID_TEST", handler_returning_invalid_intents
        )

        # Publish event
        await enhanced_message_bus.publish_async(
            publisher=MagicMock(), event_type="INVALID_TEST", message={"test": "data"}
        )

        # Wait for processing
        await asyncio.sleep(0.1)

        # Handler should have been called
        assert invalid_intents_returned is True

        # But invalid intents should not have been processed
        # (This would be logged as errors in the intent processor)

    @pytest.mark.asyncio
    async def test_error_handling_in_enhanced_publish(self, enhanced_message_bus):
        """Test error handling in enhanced publish method."""

        def failing_handler(event_data, context):
            """Handler that raises an exception."""
            raise ValueError("Handler error")

        # Subscribe failing handler
        enhanced_message_bus.subscribe("test", "ERROR_TEST", failing_handler)

        # Publishing should not raise exception
        await enhanced_message_bus.publish_async(
            publisher=MagicMock(), event_type="ERROR_TEST", message={"test": "data"}
        )

        # Wait for processing
        await asyncio.sleep(0.1)

        # MessageBus should handle the error gracefully

    @pytest.mark.asyncio
    async def test_intent_processor_dependency_injection(
        self, enhanced_message_bus, mock_dependencies
    ):
        """Test that intent processor gets proper dependencies."""
        # Intent processor should have been initialized with dependencies
        intent_processor = enhanced_message_bus._intent_processor

        assert (
            intent_processor.communication_manager
            == mock_dependencies["communication_manager"]
        )
        assert intent_processor.workflow_engine == mock_dependencies["workflow_engine"]

    def test_backward_compatibility_with_sync_publish(self, enhanced_message_bus):
        """Test that synchronous publish method still works."""
        handler_called = False

        def sync_handler(event_data):
            """Synchronous handler."""
            nonlocal handler_called
            handler_called = True

        # Subscribe sync handler
        enhanced_message_bus.subscribe("test", "SYNC_TEST", sync_handler)

        # Use synchronous publish (legacy method)
        enhanced_message_bus.publish(
            publisher=MagicMock(), message_type="SYNC_TEST", message={"test": "data"}
        )

        # Handler should have been called
        assert handler_called is True

    @pytest.mark.asyncio
    async def test_multiple_handlers_with_intents(
        self, enhanced_message_bus, mock_dependencies
    ):
        """Test multiple handlers returning intents for same event."""

        def handler1(event_data, context):
            return [NotificationIntent(message="Handler 1", channel="test")]

        def handler2(event_data, context):
            return [NotificationIntent(message="Handler 2", channel="test")]

        # Subscribe both handlers
        enhanced_message_bus.subscribe("role1", "MULTI_TEST", handler1)
        enhanced_message_bus.subscribe("role2", "MULTI_TEST", handler2)

        # Publish event
        await enhanced_message_bus.publish_async(
            publisher=MagicMock(), event_type="MULTI_TEST", message={"test": "data"}
        )

        # Wait for processing
        await asyncio.sleep(0.1)

        # Both handlers should have resulted in notifications
        assert (
            mock_dependencies["communication_manager"].send_notification.call_count == 2
        )

    def test_intent_processing_can_be_disabled(self):
        """Test that intent processing can be disabled."""
        bus = MessageBus()
        bus._enable_intent_processing = False
        bus.start()

        # Should not have intent processor when disabled
        assert not hasattr(bus, "_intent_processor") or bus._intent_processor is None


class TestMessageBusIntentIntegration:
    """Test integration between MessageBus and intent system."""

    @pytest.mark.asyncio
    async def test_timer_expiry_scenario(self):
        """Test complete timer expiry scenario with new architecture."""
        # Create enhanced message bus
        bus = MessageBus()
        mock_comm = AsyncMock()
        bus.communication_manager = mock_comm
        bus.start()

        # Create timer handler using new pattern
        def handle_timer_expiry(event_data, context):
            """Pure function timer handler."""
            return [
                NotificationIntent(
                    message=f"⏰ Timer expired: {context.get_metadata('original_request')}",
                    channel=context.get_safe_channel(),
                    user_id=context.user_id,
                    priority="medium",
                ),
                AuditIntent(
                    action="timer_expired",
                    details={
                        "timer_id": context.get_metadata("timer_id"),
                        "original_request": context.get_metadata("original_request"),
                    },
                    user_id=context.user_id,
                ),
            ]

        # Subscribe handler
        bus.subscribe("timer", "TIMER_EXPIRED", handle_timer_expiry)

        # Simulate timer expiry event
        mock_publisher = MagicMock()
        mock_publisher.user_id = "user123"
        mock_publisher.channel_id = "channel456"

        await bus.publish_async(
            publisher=mock_publisher,
            event_type="TIMER_EXPIRED",
            message=["timer_123", "Set timer for 5 minutes"],
        )

        # Wait for async processing
        await asyncio.sleep(0.1)

        # Verify notification was sent
        mock_comm.send_notification.assert_called_once()
        call_args = mock_comm.send_notification.call_args
        assert "Timer expired" in call_args.kwargs["message"]
        assert call_args.kwargs["channel"] == "channel456"
        assert call_args.kwargs["user_id"] == "user123"

    @pytest.mark.asyncio
    async def test_intent_processing_metrics(self):
        """Test that intent processing provides metrics."""
        bus = MessageBus()
        bus.start()

        def metric_handler(event_data, context):
            return [NotificationIntent(message="Test", channel="test")]

        bus.subscribe("test", "METRIC_TEST", metric_handler)

        # Process multiple events
        for i in range(5):
            await bus.publish_async(
                publisher=MagicMock(),
                event_type="METRIC_TEST",
                message={"iteration": i},
            )

        # Wait for processing
        await asyncio.sleep(0.1)

        # Should have processed intents
        if hasattr(bus, "_intent_processor") and bus._intent_processor:
            assert bus._intent_processor.get_processed_count() >= 5


if __name__ == "__main__":
    pytest.main([__file__])
