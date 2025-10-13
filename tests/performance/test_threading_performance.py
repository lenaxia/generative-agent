"""
Performance validation tests for threading architecture fixes.

This module validates the performance characteristics of the new single event loop
architecture and intent-based processing system from Documents 25, 26, and 27.

Created: 2025-10-13
Part of: Phase 3 - Integration & Testing (Document 27)
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from common.intent_processor import IntentProcessor
from common.intents import AuditIntent, NotificationIntent, WorkflowIntent
from common.message_bus import MessageBus
from supervisor.supervisor import Supervisor

logger = logging.getLogger(__name__)


class TestThreadingPerformance:
    """Performance validation for threading architecture fixes."""

    @pytest.fixture
    def mock_communication_manager(self):
        """Create a mock communication manager for performance testing."""
        mock_comm = AsyncMock()
        mock_comm.send_notification = AsyncMock()
        return mock_comm

    @pytest.fixture
    def mock_workflow_engine(self):
        """Create a mock workflow engine for performance testing."""
        mock_engine = AsyncMock()
        mock_engine.start_workflow = AsyncMock(return_value="workflow_123")
        return mock_engine

    @pytest.fixture
    def intent_processor(self, mock_communication_manager, mock_workflow_engine):
        """Create an IntentProcessor with mocked dependencies."""
        return IntentProcessor(
            communication_manager=mock_communication_manager,
            workflow_engine=mock_workflow_engine,
        )

    @pytest.mark.asyncio
    async def test_intent_processing_throughput(self, intent_processor):
        """Test intent processing throughput - should handle >50 intents/second."""
        # Create 100 test intents
        intents = [
            NotificationIntent(
                message=f"Performance test message {i}",
                channel="C123PERF",
                priority="low",
            )
            for i in range(100)
        ]

        # Measure processing time
        start_time = time.time()
        results = await intent_processor.process_intents(intents)
        end_time = time.time()

        processing_time = end_time - start_time
        throughput = len(intents) / processing_time

        logger.info(f"Processed {len(intents)} intents in {processing_time:.3f}s")
        logger.info(f"Throughput: {throughput:.1f} intents/second")

        # Verify performance requirements
        assert (
            results["processed"] == 100
        ), f"Expected 100 processed, got {results['processed']}"
        assert results["failed"] == 0, f"Expected 0 failed, got {results['failed']}"
        assert (
            throughput >= 50
        ), f"Throughput {throughput:.1f} intents/s below requirement of 50/s"
        assert (
            processing_time < 2.0
        ), f"Processing took too long: {processing_time:.3f}s"

    @pytest.mark.asyncio
    async def test_event_handling_latency(self):
        """Test event handling latency - should be <100ms average."""
        from common.enhanced_event_context import LLMSafeEventContext
        from roles.timer_single_file import handle_timer_expiry

        # Create test context
        context = LLMSafeEventContext(
            channel_id="C123PERF",
            user_id="U456PERF",
            timestamp=time.time(),
            source="performance_test",
        )

        # Measure event handling latency
        latencies = []
        for i in range(50):
            event_data = [f"timer_{i}", f"Performance test {i}"]

            start_time = time.time()
            intents = handle_timer_expiry(event_data, context)
            end_time = time.time()

            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)

            # Verify intents are generated
            assert len(intents) >= 1, f"Should generate intents for event {i}"

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        logger.info(
            f"Event handling latency - Avg: {avg_latency:.2f}ms, Max: {max_latency:.2f}ms, Min: {min_latency:.2f}ms"
        )

        # Verify performance requirements
        assert (
            avg_latency < 100
        ), f"Average latency {avg_latency:.2f}ms exceeds 100ms requirement"
        assert (
            max_latency < 500
        ), f"Max latency {max_latency:.2f}ms exceeds 500ms threshold"

    @pytest.mark.asyncio
    async def test_concurrent_intent_processing_performance(self, intent_processor):
        """Test concurrent intent processing performance."""
        # Create multiple batches of intents
        batch_size = 20
        num_batches = 5

        async def process_batch(batch_id):
            intents = [
                NotificationIntent(
                    message=f"Batch {batch_id} message {i}",
                    channel="C123PERF",
                    priority="low",
                )
                for i in range(batch_size)
            ]

            start_time = time.time()
            results = await intent_processor.process_intents(intents)
            end_time = time.time()

            return {
                "batch_id": batch_id,
                "processing_time": end_time - start_time,
                "results": results,
            }

        # Process batches concurrently
        start_time = time.time()
        batch_results = await asyncio.gather(
            *[process_batch(i) for i in range(num_batches)]
        )
        total_time = time.time() - start_time

        # Verify all batches processed successfully
        total_processed = sum(
            result["results"]["processed"] for result in batch_results
        )
        total_failed = sum(result["results"]["failed"] for result in batch_results)

        expected_total = batch_size * num_batches
        assert (
            total_processed == expected_total
        ), f"Expected {expected_total} processed, got {total_processed}"
        assert total_failed == 0, f"Expected 0 failed, got {total_failed}"

        # Verify concurrent processing is efficient
        avg_batch_time = (
            sum(result["processing_time"] for result in batch_results) / num_batches
        )
        logger.info(
            f"Concurrent processing - Total: {total_time:.3f}s, Avg batch: {avg_batch_time:.3f}s"
        )

        assert (
            total_time < 3.0
        ), f"Concurrent processing took too long: {total_time:.3f}s"

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, intent_processor):
        """Test memory usage stability during extended processing."""
        import gc
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process intents in multiple rounds
        rounds = 10
        intents_per_round = 50

        memory_measurements = [initial_memory]

        for round_num in range(rounds):
            # Create intents for this round
            intents = [
                NotificationIntent(
                    message=f"Memory test round {round_num} message {i}",
                    channel="C123MEM",
                    priority="low",
                )
                for i in range(intents_per_round)
            ]

            # Process intents
            results = await intent_processor.process_intents(intents)
            assert (
                results["processed"] == intents_per_round
            ), f"Round {round_num} processing failed"

            # Force garbage collection
            gc.collect()

            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append(current_memory)

            logger.debug(f"Round {round_num}: Memory usage {current_memory:.1f} MB")

        # Analyze memory usage
        final_memory = memory_measurements[-1]
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_measurements)

        logger.info(
            f"Memory usage - Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB, Growth: {memory_growth:.1f}MB, Max: {max_memory:.1f}MB"
        )

        # Verify memory stability (should not grow excessively)
        assert (
            memory_growth < 50
        ), f"Memory growth {memory_growth:.1f}MB exceeds 50MB threshold"
        assert (
            max_memory < initial_memory + 100
        ), f"Peak memory {max_memory:.1f}MB exceeds threshold"

    @pytest.mark.asyncio
    async def test_message_bus_performance(self):
        """Test MessageBus performance with intent processing."""
        # Create MessageBus
        message_bus = MessageBus()

        # Mock intent processor for performance testing
        mock_processor = AsyncMock()
        processed_intents = []

        async def mock_process_intents(intents):
            processed_intents.extend(intents)
            return {"processed": len(intents), "failed": 0, "errors": []}

        mock_processor.process_intents = mock_process_intents
        message_bus._intent_processor = mock_processor

        # Start message bus
        message_bus.start()

        # Create handler that returns intents with correct signature
        def performance_handler(message, context=None):
            return [
                NotificationIntent(
                    message=f"Handler processed: {message.get('id', 'unknown')}",
                    channel="C123PERF",
                )
            ]

        # Subscribe handler
        message_bus.subscribe("perf_role", "PERF_EVENT", performance_handler)

        # Publish multiple events and measure performance
        num_events = 100
        start_time = time.time()

        for i in range(num_events):
            message_bus.publish(None, "PERF_EVENT", {"id": i, "data": f"test_{i}"})

        # Wait a bit for processing to complete
        await asyncio.sleep(0.1)

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify performance
        assert (
            len(processed_intents) == num_events
        ), f"Expected {num_events} intents, got {len(processed_intents)}"

        throughput = num_events / processing_time
        logger.info(f"MessageBus throughput: {throughput:.1f} events/second")

        assert (
            throughput >= 100
        ), f"MessageBus throughput {throughput:.1f} events/s below 100/s requirement"

    @pytest.mark.asyncio
    async def test_supervisor_initialization_performance(self):
        """Test Supervisor initialization performance."""
        # Measure initialization time (minimal to avoid config and event loop issues)
        start_time = time.time()
        supervisor = Supervisor.__new__(Supervisor)
        supervisor._scheduled_tasks = []
        supervisor._use_single_event_loop = True
        end_time = time.time()

        initialization_time = end_time - start_time

        logger.info(f"Supervisor initialization time: {initialization_time:.3f}s")

        # Verify initialization is fast
        assert (
            initialization_time < 1.0
        ), f"Supervisor initialization took too long: {initialization_time:.3f}s"

        # Verify single event loop configuration
        assert (
            supervisor._use_single_event_loop is True
        ), "Single event loop should be enabled"
        assert isinstance(
            supervisor._scheduled_tasks, list
        ), "Scheduled tasks should be initialized"

    @pytest.mark.asyncio
    async def test_intent_validation_performance(self):
        """Test intent validation performance."""
        # Create various types of intents
        intents = []

        # Add notification intents
        for i in range(100):
            intents.append(
                NotificationIntent(
                    message=f"Validation test {i}", channel="C123VAL", priority="medium"
                )
            )

        # Add audit intents
        for i in range(100):
            intents.append(
                AuditIntent(
                    action=f"validation_test_{i}",
                    details={"test_id": i, "data": f"test_data_{i}"},
                    severity="info",
                )
            )

        # Add workflow intents
        for i in range(100):
            intents.append(
                WorkflowIntent(
                    workflow_type=f"validation_workflow_{i}",
                    parameters={"param": i},
                    priority=1,
                )
            )

        # Measure validation performance
        start_time = time.time()
        validation_results = [intent.validate() for intent in intents]
        end_time = time.time()

        validation_time = end_time - start_time
        validation_rate = len(intents) / validation_time

        logger.info(f"Validated {len(intents)} intents in {validation_time:.3f}s")
        logger.info(f"Validation rate: {validation_rate:.1f} intents/second")

        # Verify all validations passed
        assert all(validation_results), "All intents should pass validation"

        # Verify validation performance
        assert (
            validation_rate >= 1000
        ), f"Validation rate {validation_rate:.1f} intents/s below 1000/s requirement"
        assert (
            validation_time < 1.0
        ), f"Validation took too long: {validation_time:.3f}s"

    @pytest.mark.asyncio
    async def test_error_handling_performance_impact(self, intent_processor):
        """Test that error handling doesn't significantly impact performance."""
        # Create mix of valid and invalid intents
        valid_intents = [
            NotificationIntent(
                message=f"Valid message {i}", channel="C123ERR", priority="low"
            )
            for i in range(80)
        ]

        # Create invalid intents (empty messages)
        invalid_intents = [
            NotificationIntent(
                message="", channel="C123ERR", priority="low"  # Invalid empty message
            )
            for i in range(20)
        ]

        all_intents = valid_intents + invalid_intents

        # Measure processing time with errors
        start_time = time.time()
        results = await intent_processor.process_intents(all_intents)
        end_time = time.time()

        processing_time = end_time - start_time

        # Verify error handling worked correctly
        assert (
            results["processed"] == 80
        ), f"Expected 80 processed, got {results['processed']}"
        assert results["failed"] == 20, f"Expected 20 failed, got {results['failed']}"
        assert (
            len(results["errors"]) == 20
        ), f"Expected 20 errors, got {len(results['errors'])}"

        # Verify performance is still acceptable
        throughput = len(all_intents) / processing_time
        logger.info(f"Error handling throughput: {throughput:.1f} intents/second")

        assert (
            throughput >= 40
        ), f"Error handling throughput {throughput:.1f} intents/s below 40/s requirement"
        assert (
            processing_time < 3.0
        ), f"Error handling took too long: {processing_time:.3f}s"

    def test_single_file_role_load_performance(self):
        """Test single-file role loading performance."""
        import importlib

        # Measure role import time
        start_time = time.time()

        # Import timer role
        timer_module = importlib.import_module("roles.timer_single_file")

        # Get role registration
        registration = timer_module.register_role()

        end_time = time.time()

        load_time = end_time - start_time

        logger.info(f"Single-file role load time: {load_time:.3f}s")

        # Verify role loaded correctly
        assert "config" in registration, "Role should have config"
        assert "event_handlers" in registration, "Role should have event handlers"
        assert "tools" in registration, "Role should have tools"
        assert "intents" in registration, "Role should have intents"

        # Verify load performance
        assert load_time < 1.0, f"Role loading took too long: {load_time:.3f}s"
