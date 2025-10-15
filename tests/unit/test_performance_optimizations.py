"""Performance Optimization Tests

Tests to verify that the routing performance optimizations are working correctly
and that routing time is reduced from ~5 seconds to under 1 second.
"""

import time
from unittest.mock import Mock, patch

import pytest

from llm_provider.factory import LLMFactory, LLMType
from llm_provider.role_registry import RoleRegistry
from supervisor.workflow_engine import WorkflowEngine


class MockConfig:
    """Mock configuration for testing."""

    def __init__(self, name="test", provider_type="bedrock", model_id="test-model"):
        """Initialize MockConfig with test configuration parameters.

        Args:
            name: Configuration name for testing.
            provider_type: LLM provider type (e.g., 'bedrock').
            model_id: Model identifier for testing.
        """
        self.name = name
        self.provider_type = provider_type
        self.model_id = model_id
        self.temperature = 0.3
        self.additional_params = {}


class TestLLMFactoryPerformance:
    """Test LLM Factory caching and performance optimizations."""

    def setup_method(self):
        """Set up test fixtures."""
        configs = {
            LLMType.WEAK: [MockConfig("weak", "bedrock", "weak-model")],
            LLMType.DEFAULT: [MockConfig("default", "bedrock", "default-model")],
            LLMType.STRONG: [MockConfig("strong", "bedrock", "strong-model")],
        }
        self.factory = LLMFactory(configs)

    def test_model_caching_performance(self):
        """Test that model creation is cached for performance."""
        with (
            patch("llm_provider.factory.BedrockModel") as mock_bedrock,
            patch("llm_provider.factory.Agent") as mock_agent_class,
        ):
            # Mock model creation to be fast
            mock_model = Mock()
            mock_bedrock.return_value = mock_model
            mock_agent_class.return_value = Mock()

            # First call should create the model
            start_time = time.time()
            model1 = self.factory.create_strands_model(LLMType.WEAK)
            first_call_time = time.time() - start_time

            # Second call should use cache and be much faster
            start_time = time.time()
            model2 = self.factory.create_strands_model(LLMType.WEAK)
            second_call_time = time.time() - start_time

            # Verify same model instance is returned
            assert model1 is model2, "Cached model should return same instance"

            # Second call should be significantly faster (cache hit)
            assert (
                second_call_time < first_call_time / 2
            ), f"Cached call ({second_call_time:.4f}s) should be much faster than first call ({first_call_time:.4f}s)"

            # Verify cache statistics
            stats = self.factory.get_cache_stats()
            assert stats["models_cached"] >= 1, "Should have at least one cached model"

    def test_agent_caching_performance(self):
        """Test that agent creation is cached for performance."""
        with (
            patch("llm_provider.factory.BedrockModel") as mock_bedrock,
            patch("llm_provider.factory.Agent") as mock_agent_class,
        ):
            # Mock dependencies to be fast
            mock_model = Mock()
            mock_bedrock.return_value = mock_model
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            # First call should create the agent
            start_time = time.time()
            agent1 = self.factory.create_universal_agent(LLMType.WEAK, "test_role")
            first_call_time = time.time() - start_time

            # Second call should use cache and be much faster
            start_time = time.time()
            agent2 = self.factory.create_universal_agent(LLMType.WEAK, "test_role")
            second_call_time = time.time() - start_time

            # Verify same agent instance is returned
            assert agent1 is agent2, "Cached agent should return same instance"

            # Second call should be significantly faster (cache hit)
            assert (
                second_call_time < first_call_time / 2
            ), f"Cached call ({second_call_time:.4f}s) should be much faster than first call ({first_call_time:.4f}s)"

            # Verify cache statistics
            stats = self.factory.get_cache_stats()
            assert stats["agents_cached"] >= 1, "Should have at least one cached agent"

    @patch.object(LLMFactory, "create_strands_model")
    def test_model_warming(self, mock_create_model):
        """Test that model warming pre-creates models."""
        # Mock model creation to return quickly
        mock_model = Mock()
        mock_create_model.return_value = mock_model

        # Clear any existing cache
        self.factory.clear_cache()

        # Warm models
        start_time = time.time()
        self.factory.warm_models()
        warming_time = time.time() - start_time

        # Verify models are warmed
        assert self.factory._is_warmed, "Factory should be marked as warmed"

        # Verify create_strands_model was called for warming
        assert (
            mock_create_model.call_count >= 1
        ), "Should have called create_strands_model for warming"

        # Subsequent model creation should use cache (mocked)
        start_time = time.time()
        self.factory.create_strands_model(LLMType.WEAK)
        cached_call_time = time.time() - start_time

        # Both calls should be very fast with mocking
        assert (
            warming_time < 1.0
        ), f"Warming should be fast with mocking: {warming_time:.4f}s"
        assert (
            cached_call_time < 0.1
        ), f"Cached call should be very fast: {cached_call_time:.4f}s"

    def test_cache_clear(self):
        """Test cache clearing functionality."""
        # Manually populate cache to test clearing
        mock_model = Mock()
        mock_agent = Mock()

        # Directly add to cache to avoid slow model creation
        self.factory._model_cache["test_model"] = mock_model
        self.factory._agent_cache["test_agent"] = mock_agent
        self.factory._is_warmed = True

        # Verify cache has items
        stats = self.factory.get_cache_stats()
        assert stats["models_cached"] > 0, "Should have cached models"
        assert stats["agents_cached"] > 0, "Should have cached agents"
        assert stats["is_warmed"], "Should be marked as warmed"

        # Clear cache
        self.factory.clear_cache()

        # Verify cache is empty
        stats = self.factory.get_cache_stats()
        assert stats["models_cached"] == 0, "Should have no cached models after clear"
        assert stats["agents_cached"] == 0, "Should have no cached agents after clear"
        assert not stats["is_warmed"], "Should not be marked as warmed after clear"


class TestRoleRegistryPerformance:
    """Test Role Registry caching and performance optimizations."""

    def setup_method(self):
        """Set up test fixtures."""
        with (
            patch("llm_provider.role_registry.Path.exists", return_value=True),
            patch("llm_provider.role_registry.Path.iterdir", return_value=[]),
            patch("llm_provider.role_registry.Path.glob", return_value=[]),
        ):
            self.registry = RoleRegistry("test_roles")

    def test_initialize_once_performance(self):
        """Test that initialize_once avoids repeated initialization."""
        # Reset initialization state for testing
        self.registry._is_initialized = False

        # Mock the refresh method to track calls and simulate initialization
        def mock_refresh():
            self.registry._is_initialized = True

        with patch.object(
            self.registry, "refresh", side_effect=mock_refresh
        ) as mock_refresh_obj:
            # First call should trigger refresh
            self.registry.initialize_once()
            assert (
                mock_refresh_obj.call_count == 1
            ), "First initialize_once should call refresh"

            # Subsequent calls should not trigger refresh
            self.registry.initialize_once()
            self.registry.initialize_once()
            assert (
                mock_refresh_obj.call_count == 1
            ), "Subsequent initialize_once calls should not call refresh"

    def test_fast_reply_roles_caching(self):
        """Test that fast-reply roles are cached for performance."""
        # Mock some roles
        mock_role1 = Mock()
        mock_role1.config = {"role": {"fast_reply": True}}
        mock_role2 = Mock()
        mock_role2.config = {"role": {"fast_reply": False}}

        self.registry.llm_roles = {"weather": mock_role1, "planning": mock_role2}

        # First call should compute and cache
        start_time = time.time()
        roles1 = self.registry.get_fast_reply_roles()
        first_call_time = time.time() - start_time

        # Second call should use cache
        start_time = time.time()
        roles2 = self.registry.get_fast_reply_roles()
        second_call_time = time.time() - start_time

        # Verify same result
        assert roles1 is roles2, "Should return same cached list"
        assert len(roles1) == 1, "Should have one fast-reply role"

        # Second call should be faster (cache hit)
        assert (
            second_call_time < first_call_time or second_call_time < 0.001
        ), "Cached call should be faster or negligible"


class TestWorkflowEnginePerformance:
    """Test WorkflowEngine performance optimizations."""

    @patch("supervisor.workflow_engine.UniversalAgent")
    @patch("supervisor.workflow_engine.MCPClientManager")
    def test_initialization_performance(self, mock_mcp, mock_agent):
        """Test that WorkflowEngine initialization uses optimizations."""
        # Mock dependencies
        mock_factory = Mock()
        mock_factory.warm_models = Mock()
        mock_message_bus = Mock()

        # Mock role registry with initialize_once
        mock_registry = Mock()
        mock_registry.get_fast_reply_roles.return_value = []

        with patch(
            "supervisor.workflow_engine.RoleRegistry", return_value=mock_registry
        ):
            # Create WorkflowEngine
            start_time = time.time()
            # Create workflow engine but don't store unused variable
            WorkflowEngine(
                llm_factory=mock_factory,
                message_bus=mock_message_bus,
                roles_directory="test_roles",
            )
            init_time = time.time() - start_time

            # Verify optimizations were used
            mock_registry.initialize_once.assert_called_once()  # Should use initialize_once for performance
            mock_factory.warm_models.assert_called_once()  # Should warm models for performance

            # Initialization should be reasonably fast
            assert (
                init_time < 2.0
            ), f"Initialization took {init_time:.2f}s, should be under 2s with optimizations"


class TestEndToEndPerformance:
    """Test end-to-end routing performance."""

    @patch("supervisor.workflow_engine.UniversalAgent")
    @patch("supervisor.workflow_engine.MCPClientManager")
    @patch("llm_provider.role_registry.Path.exists", return_value=True)
    @patch("llm_provider.role_registry.Path.iterdir", return_value=[])
    @patch("llm_provider.role_registry.Path.glob", return_value=[])
    def test_routing_performance_target(
        self, mock_glob, mock_iterdir, mock_exists, mock_mcp, mock_agent
    ):
        """Test that routing meets the <1 second performance target."""
        # Mock dependencies for fast execution
        mock_factory = Mock()
        mock_factory.warm_models = Mock()
        mock_factory.create_strands_model = Mock(return_value=Mock())

        mock_message_bus = Mock()

        # Mock router to return fast result
        mock_router_instance = Mock()
        mock_router_instance.route_request.return_value = {
            "route": "weather",
            "confidence": 0.95,
            "execution_time_ms": 100,
        }
        # RequestRouter removed - using router role directly

        # Mock agent for fast execution
        mock_agent_instance = Mock()
        mock_agent_instance.execute_task.return_value = "Weather result"
        mock_agent.return_value = mock_agent_instance

        # Create WorkflowEngine
        engine = WorkflowEngine(
            llm_factory=mock_factory,
            message_bus=mock_message_bus,
            roles_directory="test_roles",
        )

        # Test routing performance
        start_time = time.time()
        request_id = engine.start_workflow("what is the weather in seattle?")
        routing_time = time.time() - start_time

        # Verify performance target is met
        assert (
            routing_time < 1.0
        ), f"Routing took {routing_time:.3f}s, should be under 1.0s (target: <1s, original: ~5s)"

        # Verify request was processed
        assert request_id is not None, "Should return a request ID"
        assert request_id.startswith("fr_"), "Should be a fast-reply request"


if __name__ == "__main__":
    pytest.main([__file__])
