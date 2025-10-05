"""
Tests for planning tools - Phase 7.2 PlanningAgent migration to @tool functions.
"""

import pytest
from unittest.mock import Mock, patch

from llm_provider.planning_tools import (
    create_task_plan,
    analyze_task_dependencies,
    validate_task_plan,
    optimize_task_plan,
    get_planning_tools,
    get_planning_tool_descriptions,
    PlanningOutput
)
from common.task_graph import TaskGraph, TaskDescription, TaskDependency


class TestPlanningTools:
    """Test planning tools functionality."""

    def test_create_task_plan_basic(self):
        """Test basic task plan creation with mock LLM factory."""
        from unittest.mock import Mock, patch
        from llm_provider.factory import LLMFactory
        
        # Create mock LLM factory
        mock_factory = Mock(spec=LLMFactory)
        mock_model = Mock()
        mock_factory.create_strands_model.return_value = mock_model
        
        # Mock the universal agent execute_task method to return JSON string
        with patch('llm_provider.universal_agent.UniversalAgent') as mock_ua_class:
            mock_ua = Mock()
            mock_ua.execute_task.return_value = '{"tasks": [{"task_id": "task_1", "task_name": "Search for Python info", "agent_id": "search", "task_type": "execution", "prompt": "Search for information about Python programming", "llm_type": "WEAK", "status": "pending"}], "dependencies": []}'
            mock_ua_class.return_value = mock_ua
            
            instruction = "Search for information about Python programming"
            result = create_task_plan(instruction, mock_factory)
            
            assert "task_graph" in result
            assert "tasks" in result
            assert "dependencies" in result
            assert "request_id" in result
        
        # Check task graph
        task_graph = result["task_graph"]
        assert isinstance(task_graph, TaskGraph)
        assert len(task_graph.nodes) == 1
        
        # Check task details - get the first (and only) task node
        task_node = list(task_graph.nodes.values())[0]
        task = result["tasks"][0]  # Get from the tasks list in result
        assert task.task_name == "Search for Python info"
        assert task.agent_id == "search"
        assert task.task_type == "execution"
        assert task.prompt == instruction

    def test_create_task_plan_with_custom_agents(self):
        """Test task plan creation with mock LLM factory."""
        from unittest.mock import Mock, patch
        from llm_provider.factory import LLMFactory
        
        # Create mock LLM factory
        mock_factory = Mock(spec=LLMFactory)
        mock_model = Mock()
        mock_factory.create_strands_model.return_value = mock_model
        
        # Mock the universal agent execute_task method to return JSON string
        with patch('llm_provider.universal_agent.UniversalAgent') as mock_ua_class:
            mock_ua = Mock()
            mock_ua.execute_task.return_value = '{"tasks": [{"task_id": "task_1", "task_name": "Get weather info", "agent_id": "weather", "task_type": "execution", "prompt": "Get weather for Seattle", "llm_type": "WEAK", "status": "pending"}], "dependencies": []}'
            mock_ua_class.return_value = mock_ua
            
            instruction = "Get weather for Seattle"
            result = create_task_plan(instruction, mock_factory, "test_request")
            
            assert result["request_id"] == "test_request"
            assert len(result["tasks"]) == 1
            
            task = result["tasks"][0]
            assert task.prompt == instruction
            assert task.agent_id == "weather"
            assert task.llm_type == "WEAK"

    def test_analyze_task_dependencies_empty(self):
        """Test dependency analysis with empty task list."""
        tasks = []
        dependencies = analyze_task_dependencies(tasks)
        
        assert dependencies == []

    def test_analyze_task_dependencies_single_task(self):
        """Test dependency analysis with single task."""
        tasks = [{"task_id": "task_1"}]
        dependencies = analyze_task_dependencies(tasks)
        
        assert dependencies == []

    def test_analyze_task_dependencies_multiple_tasks(self):
        """Test dependency analysis with multiple tasks."""
        tasks = [
            {"task_id": "task_1"},
            {"task_id": "task_2"},
            {"task_id": "task_3"}
        ]
        
        dependencies = analyze_task_dependencies(tasks)
        
        assert len(dependencies) == 2
        assert dependencies[0]["source"] == "task_1"
        assert dependencies[0]["target"] == "task_2"
        assert dependencies[1]["source"] == "task_2"
        assert dependencies[1]["target"] == "task_3"

    def test_validate_task_plan_valid(self):
        """Test validation of valid task plan."""
        tasks = [
            {
                "task_id": "task_1",
                "task_name": "Test Task",
                "agent_id": "test_agent",
                "task_type": "execution",
                "prompt": "Test prompt"
            }
        ]
        dependencies = []
        
        result = validate_task_plan(tasks, dependencies)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["task_count"] == 1
        assert result["dependency_count"] == 0

    def test_validate_task_plan_missing_fields(self):
        """Test validation with missing required fields."""
        tasks = [
            {
                "task_id": "task_1",
                "task_name": "Test Task",
                # Missing agent_id, task_type, prompt
            }
        ]
        dependencies = []
        
        result = validate_task_plan(tasks, dependencies)
        
        assert result["valid"] is False
        assert len(result["errors"]) == 3  # Missing 3 fields
        assert "agent_id" in str(result["errors"])
        assert "task_type" in str(result["errors"])
        assert "prompt" in str(result["errors"])

    def test_validate_task_plan_invalid_dependencies(self):
        """Test validation with invalid dependencies."""
        tasks = [
            {
                "task_id": "task_1",
                "task_name": "Test Task",
                "agent_id": "test_agent",
                "task_type": "execution",
                "prompt": "Test prompt"
            }
        ]
        dependencies = [
            {
                "source": "nonexistent_task",
                "target": "task_1"
            }
        ]
        
        result = validate_task_plan(tasks, dependencies)
        
        assert result["valid"] is False
        assert len(result["errors"]) == 1
        assert "nonexistent_task" in str(result["errors"])

    def test_optimize_task_plan_removes_duplicates(self):
        """Test task plan optimization removes duplicate dependencies."""
        tasks = [
            {"task_id": "task_1"},
            {"task_id": "task_2"}
        ]
        dependencies = [
            {"source": "task_1", "target": "task_2"},
            {"source": "task_1", "target": "task_2"},  # Duplicate
            {"source": "task_2", "target": "task_1"}
        ]
        
        result = optimize_task_plan(tasks, dependencies)
        
        assert len(result["dependencies"]) == 2  # Removed 1 duplicate
        assert len(result["optimizations_applied"]) > 0
        assert "Removed 1 redundant dependencies" in result["optimizations_applied"][0]

    def test_planning_output_validation_valid(self):
        """Test PlanningOutput validation with valid data."""
        task = TaskDescription(
            task_id="task_1",
            task_name="Test Task",
            agent_id="test_agent",
            task_type="execution",
            prompt="Test prompt",
            status="pending"
        )
        
        output = PlanningOutput(tasks=[task], dependencies=[])
        
        assert len(output.tasks) == 1
        assert output.dependencies == []

    def test_planning_output_validation_invalid_task(self):
        """Test PlanningOutput validation with invalid task."""
        # Create task with missing required fields
        task = TaskDescription(
            task_id="task_1",
            task_name="",  # Empty name should fail validation
            agent_id="",   # Empty agent_id should fail validation
            task_type="execution",
            prompt="Test prompt",
            status="pending"
        )
        
        with pytest.raises(ValueError, match="All tasks must have"):
            PlanningOutput(tasks=[task], dependencies=[])

    def test_planning_output_validation_invalid_dependency(self):
        """Test PlanningOutput validation with invalid dependency."""
        task = TaskDescription(
            task_id="task_1",
            task_name="Test Task",
            agent_id="test_agent",
            task_type="execution",
            prompt="Test prompt",
            status="pending"
        )
        
        # Create dependency with missing fields
        dependency = TaskDependency(
            source="",  # Empty source should fail validation
            target="task_1",
            dependency_type="sequential"
        )
        
        with pytest.raises(ValueError, match="All dependencies must have"):
            PlanningOutput(tasks=[task], dependencies=[dependency])

    def test_get_planning_tools(self):
        """Test getting all planning tools."""
        tools = get_planning_tools()
        
        assert "create_task_plan" in tools
        assert "analyze_task_dependencies" in tools
        assert "validate_task_plan" in tools
        assert "optimize_task_plan" in tools
        
        # Verify tools are callable
        assert callable(tools["create_task_plan"])
        assert callable(tools["analyze_task_dependencies"])

    def test_get_planning_tool_descriptions(self):
        """Test getting planning tool descriptions."""
        descriptions = get_planning_tool_descriptions()
        
        assert "create_task_plan" in descriptions
        assert "analyze_task_dependencies" in descriptions
        assert "validate_task_plan" in descriptions
        assert "optimize_task_plan" in descriptions
        
        # Verify descriptions are strings
        for desc in descriptions.values():
            assert isinstance(desc, str)
            assert len(desc) > 0

    @patch('llm_provider.universal_agent.UniversalAgent.execute_task')
    def test_create_task_plan_integration_with_task_graph(self, mock_execute_task):
        """Test that created task plan integrates properly with TaskGraph."""
        from unittest.mock import Mock
        from llm_provider.factory import LLMFactory
        
        # Create mock LLM factory
        mock_factory = Mock(spec=LLMFactory)
        mock_factory.get_framework.return_value = 'strands'
        
        # Mock the UniversalAgent.execute_task to return planning JSON
        mock_execute_task.return_value = '{"tasks": [{"task_id": "task_1", "task_name": "Analyze market trends", "agent_id": "analysis", "task_type": "execution", "prompt": "Analyze market trends", "llm_type": "DEFAULT", "status": "pending"}], "dependencies": []}'
        
        instruction = "Analyze market trends"
        result = create_task_plan(instruction, mock_factory, "integration_test")
        
        # Verify mock was called correctly
        mock_execute_task.assert_called_once()
        call_args = mock_execute_task.call_args
        assert call_args[1]['role'] == 'planning'
        assert call_args[1]['llm_type'].value == 'strong'
        
        task_graph = result["task_graph"]
        
        # Verify TaskGraph methods work
        assert task_graph.get_entrypoint_nodes() is not None
        assert task_graph.get_terminal_nodes() is not None
        
        # Verify task graph has correct request_id
        assert task_graph.request_id == "integration_test"
        
        # Verify llm_type is properly transferred
        node = list(task_graph.nodes.values())[0]
        assert node.llm_type == "DEFAULT"

    def test_planning_tools_no_langchain_dependencies(self):
        """Test that planning tools don't import LangChain."""
        # This test verifies that importing planning_tools doesn't require LangChain
        try:
            import llm_provider.planning_tools
            # If we get here, the import succeeded without LangChain dependencies
            assert True
        except ImportError as e:
            if 'langchain' in str(e).lower():
                pytest.fail(f"Planning tools still have LangChain dependency: {e}")
            else:
                # Some other import error, re-raise
                raise


if __name__ == "__main__":
    pytest.main([__file__])