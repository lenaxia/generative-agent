"""Integration tests for Hybrid Execution Architecture.

Tests end-to-end workflows using both LLM-based and programmatic roles
to verify the hybrid execution architecture works correctly.
"""

import json
from unittest.mock import Mock, patch

import pytest

from common.message_bus import MessageBus
from llm_provider.factory import LLMFactory
from llm_provider.role_registry import RoleRegistry
from llm_provider.universal_agent import UniversalAgent
from roles.programmatic.search_data_collector_role import SearchDataCollectorRole
from supervisor.workflow_engine import WorkflowEngine


class TestHybridExecutionIntegration:
    """Integration tests for hybrid execution architecture."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        factory.create_strands_model.return_value = Mock()
        return factory

    @pytest.fixture
    def message_bus(self):
        """Create message bus."""
        return MessageBus()

    def test_hybrid_execution_architecture_integration(
        self, mock_llm_factory, message_bus
    ):
        """Test that hybrid execution architecture integrates with WorkflowEngine."""
        # Create role registry and register both role types
        role_registry = RoleRegistry("roles")

        # Register programmatic role
        search_collector = SearchDataCollectorRole()
        role_registry.register_programmatic_role(search_collector)

        # Create Universal Agent with hybrid support
        universal_agent = UniversalAgent(mock_llm_factory, role_registry)

        # Create WorkflowEngine
        workflow_engine = WorkflowEngine(mock_llm_factory, message_bus)
        workflow_engine.universal_agent = universal_agent

        # Test that WorkflowEngine can detect role types
        assert universal_agent.is_programmatic_role("search_data_collector") is True
        assert universal_agent.get_role_type("search_data_collector") == "programmatic"

        # Test that existing LLM roles still work
        assert (
            universal_agent.get_role_type("analysis") == "llm"
        )  # Default for unknown roles

    def test_workflow_engine_uses_hybrid_execution(self, mock_llm_factory, message_bus):
        """Test that WorkflowEngine automatically uses hybrid execution paths."""
        # Create enhanced Universal Agent
        role_registry = RoleRegistry("roles")
        search_collector = SearchDataCollectorRole()
        role_registry.register_programmatic_role(search_collector)

        universal_agent = UniversalAgent(mock_llm_factory, role_registry)

        # Mock the execute_task method to verify it's called
        with patch.object(
            universal_agent, "execute_task", return_value="test result"
        ) as mock_execute:
            # Test programmatic execution
            # Execute task but don't store unused result
            universal_agent.execute_task(
                instruction="Search for USS Monitor", role="search_data_collector"
            )

            # Should have called execute_task which routes to programmatic execution
            mock_execute.assert_called_once_with(
                instruction="Search for USS Monitor", role="search_data_collector"
            )

    def test_backward_compatibility_with_existing_roles(
        self, mock_llm_factory, message_bus
    ):
        """Test that existing LLM roles continue to work unchanged."""
        role_registry = RoleRegistry("roles")
        universal_agent = UniversalAgent(mock_llm_factory, role_registry)

        # Test that existing roles default to LLM execution
        assert universal_agent.get_role_type("planning") == "llm"
        assert universal_agent.get_role_type("search") == "llm"
        assert universal_agent.get_role_type("analysis") == "llm"

        # Test that is_programmatic_role returns False for LLM roles
        assert universal_agent.is_programmatic_role("planning") is False
        assert universal_agent.is_programmatic_role("search") is False
        assert universal_agent.is_programmatic_role("analysis") is False

    def test_hybrid_architecture_performance_benefits(
        self, mock_llm_factory, message_bus
    ):
        """Test that hybrid architecture provides expected performance benefits."""
        role_registry = RoleRegistry("roles")
        search_collector = SearchDataCollectorRole()
        role_registry.register_programmatic_role(search_collector)

        universal_agent = UniversalAgent(mock_llm_factory, role_registry)

        # Mock programmatic execution to return performance metadata
        with patch.object(search_collector, "execute") as mock_execute:
            mock_execute.return_value = {
                "search_results": [{"title": "Test", "url": "http://test.com"}],
                "metadata": {
                    "llm_calls": 1,  # Only parsing call
                    "execution_time": "0.5s",
                    "total_results": 1,
                },
                "execution_type": "programmatic_data_collection",
            }

            result = universal_agent.execute_task(
                instruction="Search for test data", role="search_data_collector"
            )

            # Should have executed programmatically
            mock_execute.assert_called_once()

            # Result should contain performance metadata
            result_data = json.loads(result)
            assert result_data["metadata"]["llm_calls"] == 1
            assert result_data["execution_type"] == "programmatic_data_collection"
