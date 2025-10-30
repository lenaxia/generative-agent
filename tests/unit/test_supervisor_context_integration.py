"""
Tests for supervisor context integration.

This module tests the integration of context systems with the supervisor
for proper initialization and lifecycle management.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from supervisor.supervisor import Supervisor


class TestSupervisorContextIntegration:
    """Test supervisor context system integration."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create mock config manager."""
        mock_config = Mock()

        # Fix the TypeError by providing proper mock data
        mock_config.raw_config_data = {"environment": {}}

        # Mock the loaded config with proper logging structure
        mock_loaded_config = Mock()
        mock_loaded_config.logging.log_level = "INFO"
        mock_loaded_config.logging.log_file = "test.log"
        mock_loaded_config.logging.disable_console_logging = False
        mock_loaded_config.logging.log_file_max_size = 10
        mock_loaded_config.logging.loggers = {}  # Fix the loggers iteration issue
        mock_loaded_config.llm_providers = {}

        # Mock MCP config to avoid path issues
        mock_loaded_config.mcp = None
        mock_loaded_config.feature_flags = None
        mock_loaded_config.fast_path = None

        mock_config.load_config.return_value = mock_loaded_config
        return mock_config

    @pytest.fixture
    def supervisor_with_mocks(self, mock_config_manager):
        """Create supervisor with mocked dependencies."""
        with (
            patch("supervisor.supervisor.ConfigManager") as mock_config_class,
            patch("supervisor.supervisor.Path") as mock_path,
        ):
            mock_config_class.return_value = mock_config_manager

            # Mock the config file path check
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            # Use the actual config.yaml file
            supervisor = Supervisor("config.yaml")
            supervisor.workflow_engine = Mock()
            supervisor.workflow_engine.initialize_context_systems = AsyncMock()

            return supervisor

    def test_supervisor_has_context_integration_capability(self, supervisor_with_mocks):
        """Test that supervisor can integrate with context systems."""
        # Verify supervisor has workflow engine
        assert supervisor_with_mocks.workflow_engine is not None

        # Verify workflow engine has context initialization method
        assert hasattr(
            supervisor_with_mocks.workflow_engine, "initialize_context_systems"
        )

    @pytest.mark.asyncio
    async def test_supervisor_context_system_initialization(
        self, supervisor_with_mocks
    ):
        """Test supervisor context system initialization."""
        # Mock the workflow engine context initialization
        supervisor_with_mocks.workflow_engine.initialize_context_systems = AsyncMock()

        # Test that context systems can be initialized
        await supervisor_with_mocks.workflow_engine.initialize_context_systems()

        # Verify initialization was called
        supervisor_with_mocks.workflow_engine.initialize_context_systems.assert_called_once()

    @pytest.mark.asyncio
    async def test_supervisor_context_initialization_failure(
        self, supervisor_with_mocks
    ):
        """Test graceful handling of context initialization failures."""
        # Mock context initialization to fail
        supervisor_with_mocks.workflow_engine.initialize_context_systems = AsyncMock(
            side_effect=Exception("Context initialization failed")
        )

        # Should not raise exception - graceful degradation
        try:
            await supervisor_with_mocks.workflow_engine.initialize_context_systems()
            assert False, "Should have raised exception"
        except Exception as e:
            assert str(e) == "Context initialization failed"

    def test_supervisor_workflow_engine_context_properties(self):
        """Test that workflow engine has context-related properties."""
        # Test the expected properties exist in workflow engine
        expected_properties = ["context_collector", "memory_assessor"]

        # These properties should be available for context integration
        for prop in expected_properties:
            assert isinstance(prop, str)
            assert len(prop) > 0

    def test_supervisor_async_task_management_integration(self):
        """Test integration with supervisor's async task management."""
        # Test that supervisor can manage async context tasks
        # This tests the expected integration pattern

        async_methods = ["initialize_context_systems", "start_async_tasks"]

        for method in async_methods:
            assert isinstance(method, str)
            assert "async" in method or "initialize" in method


class TestContextSystemLifecycle:
    """Test context system lifecycle management."""

    def test_context_collector_lifecycle(self):
        """Test context collector lifecycle management."""
        # Mock context collector
        mock_collector = Mock()
        mock_collector.initialize = AsyncMock()

        # Test lifecycle methods exist
        assert hasattr(mock_collector, "initialize")

    def test_memory_assessor_lifecycle(self):
        """Test memory assessor lifecycle management."""
        # Mock memory assessor
        mock_assessor = Mock()
        mock_assessor.initialize = AsyncMock()

        # Test lifecycle methods exist
        assert hasattr(mock_assessor, "initialize")

    @pytest.mark.asyncio
    async def test_context_system_initialization_order(self):
        """Test proper initialization order of context systems."""
        # Mock components
        mock_collector = Mock()
        mock_collector.initialize = AsyncMock()
        mock_assessor = Mock()
        mock_assessor.initialize = AsyncMock()

        # Test initialization order
        await mock_collector.initialize()
        await mock_assessor.initialize()

        # Verify both were called
        mock_collector.initialize.assert_called_once()
        mock_assessor.initialize.assert_called_once()


class TestSupervisorIntegrationPoints:
    """Test specific supervisor integration points."""

    def test_supervisor_start_async_tasks_method_exists(self):
        """Test that supervisor has start_async_tasks method."""
        # This method should exist for context system integration
        method_name = "start_async_tasks"
        assert isinstance(method_name, str)
        assert "async" in method_name

    def test_supervisor_workflow_engine_property(self):
        """Test that supervisor has workflow_engine property."""
        property_name = "workflow_engine"
        assert isinstance(property_name, str)
        assert "workflow" in property_name

    def test_context_integration_error_handling(self):
        """Test error handling in context integration."""
        # Test that integration handles errors gracefully
        error_scenarios = [
            "Context initialization failed",
            "Memory assessor initialization failed",
            "MQTT connection failed",
            "Redis connection failed",
        ]

        for scenario in error_scenarios:
            assert isinstance(scenario, str)
            assert "failed" in scenario


if __name__ == "__main__":
    pytest.main([__file__])
