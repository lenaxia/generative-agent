"""Integration tests for Fast-Path Routing System.

Tests the complete fast-path routing flow from request to response,
including routing decisions, role execution, and fallback mechanisms.
"""

import time
from unittest.mock import Mock, patch

import pytest

from common.message_bus import MessageBus
from common.request_model import RequestMetadata
from llm_provider.factory import LLMFactory
from supervisor.workflow_engine import WorkflowEngine


class TestFastPathRoutingIntegration:
    """Integration tests for fast-path routing system."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Mock LLM factory for testing."""
        factory = Mock(spec=LLMFactory)

        # Mock model creation
        mock_model = Mock()
        factory.create_strands_model.return_value = mock_model

        return factory

    @pytest.fixture
    def mock_message_bus(self):
        """Mock message bus for testing."""
        return Mock(spec=MessageBus)

    @pytest.fixture
    def fast_path_config(self):
        """Fast-path configuration for testing."""
        return {
            "enabled": True,
            "confidence_threshold": 0.7,
            "max_response_time": 3000,
            "fallback_on_error": True,
            "log_routing_decisions": True,
            "track_performance_metrics": True,
        }

    @pytest.fixture
    def workflow_engine(self, mock_llm_factory, mock_message_bus, fast_path_config):
        """Create WorkflowEngine with fast-path routing enabled."""
        with patch("supervisor.workflow_engine.MCPClientManager"):
            engine = WorkflowEngine(
                llm_factory=mock_llm_factory,
                message_bus=mock_message_bus,
                fast_path_config=fast_path_config,
                roles_directory="roles",
            )
            return engine

    def test_fast_path_weather_query_integration(
        self, workflow_engine, mock_llm_factory, mock_llm_execution
    ):
        """Test complete fast-path flow for weather query."""
        # Create request
        request = RequestMetadata(
            prompt="What's the weather in Seattle?",
            source_id="test_client",
            target_id="supervisor",
        )

        # Execute request
        start_time = time.time()
        result = workflow_engine.handle_request(request)
        execution_time = time.time() - start_time

        # Verify fast-path execution - should return fast-reply ID
        assert result.startswith("fr_")  # Fast-reply ID format
        assert execution_time < 1.0  # Should be very fast in mock

        # Verify the result is stored in unified active_workflows
        assert result in workflow_engine.active_workflows
        task_context = workflow_engine.active_workflows[result]
        task_node = task_context.task_graph.nodes[result]
        assert task_node.result == "Weather data retrieved"
        assert task_node.role == "weather"
        assert task_node.task_context["confidence"] == 0.9

    def test_fast_path_calendar_query_integration(
        self, workflow_engine, mock_llm_execution
    ):
        """Test complete fast-path flow for calendar query."""
        request = RequestMetadata(
            prompt="Schedule a meeting tomorrow at 2pm",
            source_id="test_client",
            target_id="supervisor",
        )

        result = workflow_engine.handle_request(request)

        # Should return fast-reply ID
        assert result.startswith("fr_")

        # Verify the result is stored correctly
        assert result in workflow_engine.active_workflows
        task_context = workflow_engine.active_workflows[result]
        task_node = task_context.task_graph.nodes[result]
        assert task_node.result == "Task completed for role: calendar"
        assert task_node.role == "calendar"
        assert task_node.task_context["confidence"] == 0.85

    def test_fast_path_timer_query_integration(
        self, workflow_engine, mock_llm_execution
    ):
        """Test complete fast-path flow for timer query."""
        request = RequestMetadata(
            prompt="Set a timer for 10 minutes",
            source_id="test_client",
            target_id="supervisor",
        )

        result = workflow_engine.handle_request(request)

        # Should return fast-reply ID
        assert result.startswith("fr_")

        # Verify the result is stored correctly
        assert result in workflow_engine.active_workflows
        task_context = workflow_engine.active_workflows[result]
        task_node = task_context.task_graph.nodes[result]
        assert task_node.result == "Task completed for role: timer"
        assert task_node.role == "timer"
        assert task_node.task_context["confidence"] == 0.95

    def test_fast_path_smart_home_query_integration(
        self, workflow_engine, mock_llm_execution
    ):
        """Test complete fast-path flow for smart home query."""
        request = RequestMetadata(
            prompt="Turn off the living room lights",
            source_id="test_client",
            target_id="supervisor",
        )

        result = workflow_engine.handle_request(request)

        # Should return fast-reply ID
        assert result.startswith("fr_")

        # Verify the result is stored correctly
        assert result in workflow_engine.active_workflows
        task_context = workflow_engine.active_workflows[result]
        task_node = task_context.task_graph.nodes[result]
        assert task_node.result == "Task completed for role: smart_home"
        assert task_node.role == "smart_home"
        assert task_node.task_context["confidence"] == 0.88

    def test_complex_workflow_fallback_integration(
        self, workflow_engine, mock_planning_phase
    ):
        """Test fallback to complex workflow for complex requests."""
        request = RequestMetadata(
            prompt="Create a comprehensive project plan with multiple phases and dependencies",
            source_id="test_client",
            target_id="supervisor",
        )

        result = workflow_engine.handle_request(request)

        # Should return workflow ID (not fast-reply ID) due to PLANNING route
        assert result.startswith("wf_")
        # Complex workflows are stored in active_workflows but with different structure
        assert result in workflow_engine.active_workflows

    def test_low_confidence_fallback_integration(
        self, workflow_engine, mock_planning_phase
    ):
        """Test fallback to complex workflow for low confidence routing."""
        request = RequestMetadata(
            prompt="Maybe something weather related?",
            source_id="test_client",
            target_id="supervisor",
        )

        result = workflow_engine.handle_request(request)

        # Should fallback to complex workflow due to low confidence (0.3 < 0.7 threshold)
        assert result.startswith("wf_")
        # Complex workflows are stored in active_workflows
        assert result in workflow_engine.active_workflows

    def test_fast_path_error_fallback_integration(
        self, workflow_engine, mock_planning_phase
    ):
        """Test fallback to complex workflow when fast-path execution fails."""
        # Mock Universal Agent to raise an exception for weather role
        with patch.object(
            workflow_engine.universal_agent, "execute_task"
        ) as mock_execute:

            def side_effect(instruction, role, llm_type, context=None):
                if role == "router":
                    return '{"route": "weather", "confidence": 0.9}'
                elif role == "weather":
                    raise Exception("Weather service unavailable")
                else:
                    return "fallback response"

            mock_execute.side_effect = side_effect

            request = RequestMetadata(
                prompt="What's the weather?",
                source_id="test_client",
                target_id="supervisor",
            )

            result = workflow_engine.handle_request(request)

            # Should fallback to complex workflow due to execution error
            assert result.startswith("wf_")
            # Complex workflows are stored in active_workflows
            assert result in workflow_engine.active_workflows

    def test_fast_path_disabled_integration(
        self, mock_llm_factory, mock_message_bus, mock_planning_phase
    ):
        """Test behavior when fast-path routing is disabled."""
        fast_path_config = {
            "enabled": False,
            "confidence_threshold": 0.7,
            "max_response_time": 3000,
            "fallback_on_error": True,
        }

        with patch("supervisor.workflow_engine.MCPClientManager"):
            engine = WorkflowEngine(
                llm_factory=mock_llm_factory,
                message_bus=mock_message_bus,
                fast_path_config=fast_path_config,
                roles_directory="roles",
            )

            request = RequestMetadata(
                prompt="What's the weather?",
                source_id="test_client",
                target_id="supervisor",
            )

            result = engine.handle_request(request)

            # Should always use complex workflow when fast-path is disabled
            assert result.startswith("wf_")
            # Complex workflows are stored in active_workflows
            assert result in engine.active_workflows


class TestFastPathRoutingPerformance:
    """Performance tests for fast-path routing."""

    @pytest.fixture
    def workflow_engine_perf(self):
        """Create WorkflowEngine for performance testing."""
        mock_llm_factory = Mock(spec=LLMFactory)
        mock_model = Mock()
        mock_llm_factory.create_strands_model.return_value = mock_model

        mock_message_bus = Mock(spec=MessageBus)

        fast_path_config = {
            "enabled": True,
            "confidence_threshold": 0.7,
            "max_response_time": 3000,
            "fallback_on_error": True,
        }

        with patch("supervisor.workflow_engine.MCPClientManager"):
            engine = WorkflowEngine(
                llm_factory=mock_llm_factory,
                message_bus=mock_message_bus,
                fast_path_config=fast_path_config,
                roles_directory="roles",
            )
            return engine

    def test_fast_path_response_time_benchmark(
        self, workflow_engine_perf, mock_llm_execution
    ):
        """Benchmark fast-path response times."""
        # Measure multiple requests
        response_times = []
        results = []

        for i in range(10):
            request = RequestMetadata(
                prompt=f"What's the weather? {i}",
                source_id="perf_test",
                target_id="supervisor",
            )

            start_time = time.time()
            result = workflow_engine_perf.handle_request(request)
            end_time = time.time()

            response_times.append(end_time - start_time)
            results.append(result)

            # Verify it returns a workflow ID (fast-path routing falls back to workflow)
            assert result.startswith("wf_")

        # Verify performance characteristics
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)

        # In mock environment, should be very fast
        assert avg_response_time < 0.1  # 100ms average
        assert max_response_time < 0.2  # 200ms max

        print(f"Average response time: {avg_response_time*1000:.1f}ms")
        print(f"Max response time: {max_response_time*1000:.1f}ms")

    def test_concurrent_fast_path_requests(
        self, workflow_engine_perf, mock_llm_execution
    ):
        """Test concurrent fast-path request handling."""
        import concurrent.futures

        def make_request(request_id):
            request = RequestMetadata(
                prompt=f"Weather request {request_id}",
                source_id=f"client_{request_id}",
                target_id="supervisor",
            )

            start_time = time.time()
            result = workflow_engine_perf.handle_request(request)
            end_time = time.time()

            return result, end_time - start_time

        # Execute concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # Verify all requests completed successfully
        assert len(results) == 10
        for result, response_time in results:
            assert result.startswith(
                "wf_"
            )  # Workflow ID (fast-path falls back to workflow)
            assert response_time < 0.2  # Should be fast even under concurrency


class TestFastPathRoutingEdgeCases:
    """Test edge cases and error conditions."""

    def test_malformed_routing_response_integration(
        self, mock_llm_execution, mock_planning_phase
    ):
        """Test handling of malformed routing responses."""
        mock_llm_factory = Mock(spec=LLMFactory)
        mock_model = Mock()
        mock_llm_factory.create_strands_model.return_value = mock_model

        mock_message_bus = Mock(spec=MessageBus)

        fast_path_config = {"enabled": True, "confidence_threshold": 0.7}

        with patch("supervisor.workflow_engine.MCPClientManager"):
            workflow_engine = WorkflowEngine(
                llm_factory=mock_llm_factory,
                message_bus=mock_message_bus,
                fast_path_config=fast_path_config,
                roles_directory="roles",
            )

            # Mock the universal agent to return invalid JSON for router role
            with patch.object(
                workflow_engine.universal_agent, "execute_task"
            ) as mock_execute:

                def side_effect(instruction, role, llm_type, context=None):
                    if role == "router":
                        return "invalid json response"  # Malformed JSON
                    else:
                        return "fallback response"

                mock_execute.side_effect = side_effect

                request = RequestMetadata(
                    prompt="What's the weather?",
                    source_id="test_client",
                    target_id="supervisor",
                )

                result = workflow_engine.handle_request(request)

                # Should fallback to complex workflow due to malformed routing response
                assert result.startswith("wf_")
                # Complex workflows are stored in active_workflows
                assert result in workflow_engine.active_workflows

    def test_unknown_role_routing_integration(
        self, mock_llm_execution, mock_planning_phase
    ):
        """Test handling of routing to unknown roles."""
        mock_llm_factory = Mock(spec=LLMFactory)
        mock_model = Mock()
        mock_llm_factory.create_strands_model.return_value = mock_model

        mock_message_bus = Mock(spec=MessageBus)

        fast_path_config = {"enabled": True, "confidence_threshold": 0.7}

        with patch("supervisor.workflow_engine.MCPClientManager"):
            workflow_engine = WorkflowEngine(
                llm_factory=mock_llm_factory,
                message_bus=mock_message_bus,
                fast_path_config=fast_path_config,
                roles_directory="roles",
            )

            # Mock the universal agent to return unknown role
            with patch.object(
                workflow_engine.universal_agent, "execute_task"
            ) as mock_execute:

                def side_effect(instruction, role, llm_type, context=None):
                    if role == "router":
                        return '{"route": "unknown_role", "confidence": 0.9}'
                    elif role == "unknown_role":
                        raise Exception("Unknown role: unknown_role")
                    else:
                        return "fallback response"

                mock_execute.side_effect = side_effect

                request = RequestMetadata(
                    prompt="Do something with unknown role",
                    source_id="test_client",
                    target_id="supervisor",
                )

                result = workflow_engine.handle_request(request)

                # Should fallback to complex workflow due to unknown role error
                assert result.startswith("wf_")
                # Complex workflows are stored in active_workflows
                assert result in workflow_engine.active_workflows
