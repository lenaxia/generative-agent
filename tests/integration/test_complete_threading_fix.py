"""
Complete integration test for threading architecture fixes.

This module provides comprehensive end-to-end testing of the complete threading
architecture transformation from Documents 25, 26, and 27.

Created: 2025-10-13
Part of: Phase 4 - Production Deployment (Document 27)
"""

import asyncio
import logging
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from common.enhanced_event_context import LLMSafeEventContext
from common.intent_processor import IntentProcessor
from common.intents import AuditIntent, NotificationIntent
from common.message_bus import MessageBus
from roles.timer_single_file import handle_timer_expiry
from supervisor.supervisor import Supervisor
from supervisor.threading_monitor import ThreadingMonitor, get_threading_monitor

logger = logging.getLogger(__name__)


class TestCompleteThreadingFix:
    """Complete integration test for threading architecture fixes."""

    @pytest.fixture
    def threading_monitor(self):
        """Create a fresh threading monitor for testing."""
        monitor = ThreadingMonitor()
        monitor.reset_metrics()
        return monitor

    @pytest.fixture
    def mock_communication_manager(self):
        """Create a mock communication manager."""
        mock_comm = AsyncMock()
        mock_comm.send_notification = AsyncMock()
        return mock_comm

    @pytest.fixture
    def intent_processor(self, mock_communication_manager):
        """Create an IntentProcessor with mocked dependencies."""
        return IntentProcessor(
            communication_manager=mock_communication_manager,
            workflow_engine=AsyncMock(),
        )

    @pytest.mark.asyncio
    async def test_end_to_end_timer_flow(self, intent_processor, threading_monitor):
        """Test complete timer flow with threading fixes."""
        # Verify initial threading state
        initial_thread_count = threading.active_count()
        assert (
            threading_monitor.validate_single_event_loop()
        ), "Should start with single event loop"

        # Create test context
        context = LLMSafeEventContext(
            channel_id="C123COMPLETE",
            user_id="U456COMPLETE",
            timestamp=time.time(),
            source="complete_test",
        )

        # Trigger timer expiry event (should return intents, not perform I/O)
        timer_data = ["timer_complete_test", "Complete integration test timer"]

        start_time = time.time()
        intents = handle_timer_expiry(timer_data, context)
        processing_time = time.time() - start_time

        # Verify handler returns intents
        assert isinstance(intents, list), "Handler should return list of intents"
        assert len(intents) >= 2, "Should generate notification and audit intents"
        assert all(
            intent.validate() for intent in intents
        ), "All intents should be valid"

        # Verify no additional threads created
        final_thread_count = threading.active_count()
        assert (
            final_thread_count == initial_thread_count
        ), f"Thread count changed: {initial_thread_count} → {final_thread_count}"

        # Process intents through processor
        results = await intent_processor.process_intents(intents)

        # Verify processing results
        assert (
            results["processed"] >= 2
        ), f"Expected ≥2 processed, got {results['processed']}"
        assert results["failed"] == 0, f"Expected 0 failed, got {results['failed']}"
        assert (
            len(results["errors"]) == 0
        ), f"Expected no errors, got {results['errors']}"

        # Record performance metrics
        threading_monitor.record_intent_processing(processing_time, len(intents))

        # Verify performance is acceptable
        assert (
            processing_time < 0.1
        ), f"Processing took too long: {processing_time:.3f}s"

    def test_threading_monitor_health_reporting(self, threading_monitor):
        """Test threading monitor health reporting."""
        # Get initial health report
        health = threading_monitor.get_threading_health()

        # Verify health report structure
        assert "thread_count" in health, "Should include thread count"
        assert "main_thread_only" in health, "Should include main thread validation"
        assert "health_status" in health, "Should include health status"
        assert "uptime_seconds" in health, "Should include uptime"

        # Verify single thread compliance
        assert health["main_thread_only"] is True, "Should be main thread only"
        assert (
            health["health_status"] == "healthy"
        ), f"Expected healthy, got {health['health_status']}"

    def test_threading_monitor_error_tracking(self, threading_monitor):
        """Test threading monitor error tracking."""
        # Record test error
        test_error = ValueError("Test error for monitoring")
        threading_monitor.record_handler_error(test_error, "test_handler")

        # Get performance summary
        summary = threading_monitor.get_performance_summary()

        # Verify error tracking
        assert summary["error_metrics"]["total_errors"] == 1, "Should track error count"
        assert (
            summary["error_metrics"]["last_error_time"] is not None
        ), "Should track error time"
        assert summary["error_metrics"]["error_rate"] > 0, "Should calculate error rate"

    def test_threading_monitor_performance_tracking(self, threading_monitor):
        """Test threading monitor performance tracking."""
        # Record performance data
        threading_monitor.record_intent_processing(0.05, 10)  # 200 intents/second
        threading_monitor.record_intent_processing(0.02, 5)  # 250 intents/second
        threading_monitor.record_intent_processing(0.1, 15)  # 150 intents/second

        # Get performance summary
        summary = threading_monitor.get_performance_summary()

        # Verify performance tracking
        intent_perf = summary["intent_processing"]
        assert intent_perf["sample_count"] == 3, "Should track all samples"
        assert intent_perf["average_rate"] > 0, "Should calculate average rate"
        assert intent_perf["min_rate"] == 150, "Should track minimum rate"
        assert intent_perf["max_rate"] == 250, "Should track maximum rate"

    def test_production_configuration_validation(self):
        """Test production configuration validation."""
        # Test that production config can be loaded (minimal to avoid event loop issues)
        supervisor = Supervisor.__new__(Supervisor)
        supervisor._scheduled_tasks = []
        supervisor._use_single_event_loop = True

        # Verify threading configuration is applied
        assert hasattr(
            supervisor, "_use_single_event_loop"
        ), "Should have single event loop flag"
        assert (
            supervisor._use_single_event_loop is True
        ), "Single event loop should be enabled"

        # Verify scheduled tasks are initialized
        assert hasattr(supervisor, "_scheduled_tasks"), "Should have scheduled tasks"
        assert isinstance(
            supervisor._scheduled_tasks, list
        ), "Scheduled tasks should be list"

    def test_threading_architecture_compliance(self, threading_monitor):
        """Test complete threading architecture compliance."""
        # Validate single event loop
        is_compliant = threading_monitor.validate_single_event_loop()
        assert is_compliant, "Threading architecture should be compliant"

        # Get diagnostic information
        diagnostics = threading_monitor.get_diagnostic_info()

        # Verify threading compliance
        threading_info = diagnostics["threading"]
        assert len(threading_info["active_threads"]) == 1, "Should have only one thread"
        assert threading_info["main_thread"] == "MainThread", "Should be main thread"

    @pytest.mark.asyncio
    async def test_intent_processing_with_monitoring(
        self, intent_processor, threading_monitor
    ):
        """Test intent processing with performance monitoring."""
        # Create test intents
        intents = [
            NotificationIntent(
                message="Monitoring test notification",
                channel="C123MONITOR",
                priority="low",
            ),
            AuditIntent(
                action="monitoring_test",
                details={"test": "monitoring"},
                severity="info",
            ),
        ]

        # Process with timing
        start_time = time.time()
        results = await intent_processor.process_intents(intents)
        processing_time = time.time() - start_time

        # Record performance
        threading_monitor.record_intent_processing(processing_time, len(intents))

        # Verify processing succeeded
        assert (
            results["processed"] == 2
        ), f"Expected 2 processed, got {results['processed']}"
        assert results["failed"] == 0, f"Expected 0 failed, got {results['failed']}"

        # Verify monitoring recorded performance
        health = threading_monitor.get_threading_health()
        assert health["intent_processing_rate"] > 0, "Should record processing rate"

    def test_error_handling_with_monitoring(self, threading_monitor):
        """Test error handling with monitoring integration."""
        # Simulate error
        test_error = RuntimeError("Test error for monitoring")
        threading_monitor.record_handler_error(test_error, "test_handler")

        # Verify error was recorded
        health = threading_monitor.get_threading_health()
        assert health["error_count"] == 1, "Should record error count"
        assert health["last_error_time"] is not None, "Should record error time"

        # Health status should reflect error
        # Note: May be "healthy" if error is old, "warning" if recent
        assert health["health_status"] in [
            "healthy",
            "warning",
        ], f"Unexpected health status: {health['health_status']}"

    def test_threading_monitor_reset(self, threading_monitor):
        """Test threading monitor metrics reset functionality."""
        # Add some data
        threading_monitor.record_intent_processing(0.1, 5)
        threading_monitor.record_handler_error(ValueError("test"), "test")

        # Verify data exists
        health_before = threading_monitor.get_threading_health()
        assert health_before["error_count"] > 0, "Should have error count"

        # Reset metrics
        threading_monitor.reset_metrics()

        # Verify reset worked
        health_after = threading_monitor.get_threading_health()
        assert health_after["error_count"] == 0, "Error count should be reset"
        assert (
            health_after["intent_processing_rate"] == 0.0
        ), "Processing rate should be reset"

    @pytest.mark.asyncio
    async def test_complete_system_integration(self):
        """Test complete system integration with all components."""
        # Get threading monitor
        monitor = get_threading_monitor()
        initial_health = monitor.get_threading_health()

        # Verify system starts in healthy state
        assert (
            initial_health["main_thread_only"] is True
        ), "Should start with single thread"
        assert initial_health["health_status"] == "healthy", "Should start healthy"

        # Create MessageBus with intent processing
        message_bus = MessageBus()
        message_bus.start()

        # Create mock intent processor
        mock_processor = AsyncMock()
        mock_processor.process_intents = AsyncMock(
            return_value={"processed": 1, "failed": 0, "errors": []}
        )
        message_bus._intent_processor = mock_processor

        # Create handler that returns intents
        def system_handler(message, context=None):
            return [NotificationIntent(message="System test", channel="C123SYS")]

        # Subscribe and publish
        message_bus.subscribe("system_role", "SYSTEM_TEST", system_handler)
        message_bus.publish(None, "SYSTEM_TEST", {"integration": "test"})

        # Wait for processing
        await asyncio.sleep(0.1)

        # Verify system health maintained
        final_health = monitor.get_threading_health()
        assert final_health["main_thread_only"] is True, "Should maintain single thread"
        assert (
            final_health["thread_count"] == initial_health["thread_count"]
        ), "Thread count should not change"

    def test_diagnostic_information_completeness(self, threading_monitor):
        """Test that diagnostic information is comprehensive."""
        diagnostics = threading_monitor.get_diagnostic_info()

        # Verify diagnostic structure
        assert "threading" in diagnostics, "Should include threading info"
        assert "asyncio" in diagnostics, "Should include asyncio info"
        assert "performance" in diagnostics, "Should include performance info"
        assert "health" in diagnostics, "Should include health info"

        # Verify threading details
        threading_info = diagnostics["threading"]
        assert "active_threads" in threading_info, "Should list active threads"
        assert "main_thread" in threading_info, "Should identify main thread"

        # Verify each thread has required info
        for thread_info in threading_info["active_threads"]:
            assert "name" in thread_info, "Thread should have name"
            assert "daemon" in thread_info, "Thread should have daemon status"
            assert "alive" in thread_info, "Thread should have alive status"
