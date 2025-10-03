"""
Unit tests for dynamic role generation functionality in UniversalAgent.

Tests the new capability to dynamically generate roles when "None" is specified
or when a requested role is not found in the registry.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from llm_provider.universal_agent import UniversalAgent
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.role_registry import RoleRegistry
from common.task_context import TaskContext
from common.task_graph import TaskGraph, TaskDescription


class TestDynamicRoleGeneration:
    """Test suite for dynamic role generation functionality."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        mock_model = Mock()
        mock_model.invoke.return_value = Mock(content='{"selected_tools": ["search_tools", "data_processing"], "system_prompt": "You are a specialized search and data processing agent."}')
        factory.create_strands_model.return_value = mock_model
        return factory

    @pytest.fixture
    def mock_role_registry(self):
        """Create a mock role registry with shared tools."""
        registry = Mock(spec=RoleRegistry)
        
        # Mock shared tools
        mock_search_tool = Mock()
        mock_search_tool.__name__ = "search_tool"
        mock_search_tool.__doc__ = "Search for information on the web"
        
        mock_data_tool = Mock()
        mock_data_tool.__name__ = "data_processing"
        mock_data_tool.__doc__ = "Process and analyze data"
        
        registry.get_all_shared_tools.return_value = {
            "search_tools": mock_search_tool,
            "data_processing": mock_data_tool,
            "weather_tools": Mock(__name__="weather_tool", __doc__="Get weather information"),
            "slack_tools": Mock(__name__="slack_tool", __doc__="Send Slack messages")
        }
        
        # Mock role lookup (return None for unknown roles)
        registry.get_role.return_value = None
        
        return registry

    @pytest.fixture
    def mock_task_context(self):
        """Create a mock task context with task information."""
        context = Mock(spec=TaskContext)
        
        # Mock task graph with pending task
        mock_task_graph = Mock()
        mock_task_node = Mock()
        mock_task_node.task_name = "Search for weather data"
        mock_task_node.prompt = "Find current weather information for Seattle"
        mock_task_node.task_type = "search"
        mock_task_node.status = Mock(value='PENDING')
        
        mock_task_graph.nodes = {"task_1": mock_task_node}
        context.task_graph = mock_task_graph
        
        return context

    @pytest.fixture
    def universal_agent(self, mock_llm_factory, mock_role_registry):
        """Create UniversalAgent instance with mocked dependencies."""
        return UniversalAgent(
            llm_factory=mock_llm_factory,
            role_registry=mock_role_registry
        )

    def test_assume_role_with_none_triggers_dynamic_generation(self, universal_agent, mock_task_context):
        """Test that role="None" triggers dynamic role generation."""
        # Test with role="None"
        result = universal_agent.assume_role("None", LLMType.DEFAULT, mock_task_context)
        
        # Verify dynamic role generation was triggered
        assert result is not None
        assert universal_agent.current_role == "dynamic_generated"
        assert universal_agent.current_llm_type == LLMType.DEFAULT

    def test_assume_role_with_missing_role_triggers_dynamic_generation(self, universal_agent, mock_task_context):
        """Test that missing role triggers dynamic role generation."""
        # Test with non-existent role
        result = universal_agent.assume_role("non_existent_role", LLMType.DEFAULT, mock_task_context)
        
        # Verify dynamic role generation was triggered
        assert result is not None
        assert universal_agent.current_role == "dynamic_non_existent_role"
        assert universal_agent.current_llm_type == LLMType.DEFAULT

    def test_dynamic_role_generation_with_task_context(self, universal_agent, mock_task_context):
        """Test dynamic role generation extracts task information correctly."""
        with patch('strands.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            # Call dynamic role generation
            universal_agent.assume_role("None", LLMType.DEFAULT, mock_task_context)
            
            # Verify LLM was called for role generation
            universal_agent.llm_factory.create_strands_model.assert_called_with(LLMType.DEFAULT)
            
            # Verify model.invoke was called with task information
            mock_model = universal_agent.llm_factory.create_strands_model.return_value
            mock_model.invoke.assert_called_once()
            
            # Check that the prompt contains task information
            call_args = mock_model.invoke.call_args[0][0]
            assert "Search for weather data" in call_args
            assert "Find current weather information for Seattle" in call_args

    def test_dynamic_role_generation_tool_selection(self, universal_agent, mock_task_context):
        """Test that dynamic role generation selects appropriate tools."""
        # Call dynamic role generation
        result = universal_agent.assume_role("None", LLMType.DEFAULT, mock_task_context)
        
        # Verify agent was created successfully
        assert result is not None
        assert universal_agent.current_role == "dynamic_generated"
        
        # Verify LLM was called for tool selection
        universal_agent.llm_factory.create_strands_model.assert_called_with(LLMType.DEFAULT)

    def test_dynamic_role_generation_custom_system_prompt(self, universal_agent, mock_task_context):
        """Test that dynamic role generation creates custom system prompt."""
        # Call dynamic role generation
        result = universal_agent.assume_role("None", LLMType.DEFAULT, mock_task_context)
        
        # Verify agent was created successfully
        assert result is not None
        assert universal_agent.current_role == "dynamic_generated"
        
        # Verify LLM was called for prompt generation
        mock_model = universal_agent.llm_factory.create_strands_model.return_value
        mock_model.invoke.assert_called_once()

    def test_dynamic_role_generation_without_task_context(self, universal_agent):
        """Test dynamic role generation works without task context."""
        # Call without task context
        result = universal_agent.assume_role("None", LLMType.DEFAULT, None)
        
        # Should still work and create an agent
        assert result is not None
        assert universal_agent.current_role == "dynamic_generated"

    def test_dynamic_role_generation_fallback_on_llm_failure(self, universal_agent, mock_task_context):
        """Test fallback behavior when LLM role generation fails."""
        # Make LLM call fail
        universal_agent.llm_factory.create_strands_model.return_value.invoke.side_effect = Exception("LLM failed")
        
        # Call dynamic role generation
        result = universal_agent.assume_role("None", LLMType.DEFAULT, mock_task_context)
        
        # Should still create an agent with fallback configuration
        assert result is not None
        # Should use fallback configuration
        assert universal_agent.current_role == "dynamic_generated"

    def test_dynamic_role_generation_without_shared_tools(self, mock_llm_factory):
        """Test dynamic role generation when no shared tools are available."""
        # Create registry with no shared tools
        empty_registry = Mock(spec=RoleRegistry)
        empty_registry.get_all_shared_tools.return_value = {}
        empty_registry.get_role.return_value = None
        
        agent = UniversalAgent(llm_factory=mock_llm_factory, role_registry=empty_registry)
        
        # Call dynamic role generation
        result = agent.assume_role("None", LLMType.DEFAULT, None)
        
        # Should create basic dynamic agent
        assert result is not None
        assert agent.current_role == "dynamic_basic"

    def test_dynamic_role_generation_without_llm_factory(self, mock_role_registry):
        """Test dynamic role generation when no LLM factory is available."""
        agent = UniversalAgent(llm_factory=None, role_registry=mock_role_registry)
        
        # Call dynamic role generation
        result = agent.assume_role("None", LLMType.DEFAULT, None)
        
        # Should create basic dynamic agent
        assert result is not None
        assert agent.current_role == "dynamic_basic"

    def test_extract_task_info_with_suggested_role(self, universal_agent, mock_task_context):
        """Test task info extraction includes suggested role."""
        task_info = universal_agent._extract_task_info(mock_task_context, "custom_analyzer")
        
        assert "Suggested role context: custom_analyzer" in task_info
        assert "Current task: Search for weather data" in task_info
        assert "Task prompt: Find current weather information for Seattle" in task_info

    def test_extract_task_info_without_context(self, universal_agent):
        """Test task info extraction without context."""
        task_info = universal_agent._extract_task_info(None, "test_role")
        
        assert task_info == "Suggested role context: test_role"

    def test_extract_task_info_minimal(self, universal_agent):
        """Test task info extraction with minimal information."""
        task_info = universal_agent._extract_task_info(None, None)
        
        assert task_info == "General assistance task"

    def test_get_selected_tools_filters_valid_tools(self, universal_agent):
        """Test that get_selected_tools only returns valid tools."""
        available_tools = {
            "tool1": Mock(),
            "tool2": Mock(),
            "tool3": Mock()
        }
        
        # Request some valid and some invalid tools
        requested_tools = ["tool1", "invalid_tool", "tool3", "another_invalid"]
        
        result = universal_agent._get_selected_tools(requested_tools, available_tools)
        
        # Should only return valid tools
        assert len(result) == 2
        assert available_tools["tool1"] in result
        assert available_tools["tool3"] in result

    def test_generate_dynamic_role_config_json_parsing(self, universal_agent):
        """Test that dynamic role config handles JSON parsing correctly."""
        available_tools = {"test_tool": Mock(__doc__="Test tool for testing")}
        
        # Mock successful JSON response
        with patch.object(universal_agent.llm_factory, 'create_strands_model') as mock_create:
            mock_model = Mock()
            mock_model.invoke.return_value = Mock(content='{"selected_tools": ["test_tool"], "system_prompt": "Test prompt"}')
            mock_create.return_value = mock_model
            
            result = universal_agent._generate_dynamic_role_config("test task", available_tools)
            
            assert result["selected_tools"] == ["test_tool"]
            assert result["system_prompt"] == "Test prompt"

    def test_generate_dynamic_role_config_invalid_json_fallback(self, universal_agent):
        """Test fallback when LLM returns invalid JSON."""
        available_tools = {"test_tool": Mock(__doc__="Test tool")}
        
        # Mock invalid JSON response
        with patch.object(universal_agent.llm_factory, 'create_strands_model') as mock_create:
            mock_model = Mock()
            mock_model.invoke.return_value = Mock(content='invalid json')
            mock_create.return_value = mock_model
            
            result = universal_agent._generate_dynamic_role_config("test task", available_tools)
            
            # Should fallback to basic configuration
            assert "selected_tools" in result
            assert "system_prompt" in result
            assert len(result["selected_tools"]) <= 3  # Fallback limits to 3 tools