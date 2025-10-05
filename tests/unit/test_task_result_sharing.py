"""
Unit tests for task result sharing functionality.

Tests the enhanced WorkflowEngine capability to pass predecessor task results
to dependent tasks, eliminating duplicate work and improving efficiency.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from supervisor.workflow_engine import WorkflowEngine, TaskPriority
from common.task_context import TaskContext, ExecutionState
from common.task_graph import TaskGraph, TaskDescription, TaskDependency, TaskNode, TaskStatus
from common.message_bus import MessageBus
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.universal_agent import UniversalAgent


class TestTaskResultSharing:
    """Test task result sharing between dependent tasks."""
    
    @pytest.fixture
    def mock_llm_factory(self):
        """Create mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        return factory
    
    @pytest.fixture
    def mock_message_bus(self):
        """Create mock message bus."""
        return Mock(spec=MessageBus)
    
    @pytest.fixture
    def mock_universal_agent(self):
        """Create mock universal agent."""
        agent = Mock(spec=UniversalAgent)
        return agent
    
    @pytest.fixture
    def workflow_engine(self, mock_llm_factory, mock_message_bus, mock_universal_agent):
        """Create WorkflowEngine with mocked dependencies."""
        engine = WorkflowEngine(
            llm_factory=mock_llm_factory,
            message_bus=mock_message_bus,
            max_concurrent_tasks=2,
            checkpoint_interval=60
        )
        engine.universal_agent = mock_universal_agent
        return engine

    def test_dependent_task_receives_predecessor_results(self, workflow_engine, mock_universal_agent):
        """Test that dependent tasks receive predecessor results in their prompt."""
        # Create search -> analysis workflow
        tasks = [
            TaskDescription(
                task_name="Search for USS Monitor information",
                agent_id="search",
                task_type="search",
                prompt="Search for comprehensive information about USS Monitor",
                llm_type="WEAK"
            ),
            TaskDescription(
                task_name="Analyze USS Monitor information",
                agent_id="analysis", 
                task_type="analysis",
                prompt="Analyze the retrieved information about USS Monitor",
                llm_type="DEFAULT"
            )
        ]
        
        dependencies = [
            TaskDependency(
                source="Search for USS Monitor information",
                target="Analyze USS Monitor information"
            )
        ]
        
        task_context = TaskContext.from_tasks(tasks, dependencies, request_id="test_request")
        
        # Complete the search task first
        search_task = None
        analysis_task = None
        
        for task in task_context.task_graph.nodes.values():
            if task.agent_id == "search":
                search_task = task
            elif task.agent_id == "analysis":
                analysis_task = task
        
        # Complete search task with results
        search_result = "USS Monitor search results: Revolutionary Civil War ironclad warship designed by John Ericsson"
        task_context.complete_task(search_task.task_id, search_result)
        
        # Mock the universal agent for the analysis task
        mock_universal_agent.execute_task.return_value = "Analysis complete"
        
        # Execute the analysis task
        workflow_engine.delegate_task(task_context, analysis_task)
        
        # Verify the universal agent was called
        mock_universal_agent.execute_task.assert_called_once()
        call_args = mock_universal_agent.execute_task.call_args
        
        # The instruction should include predecessor results
        instruction = call_args.kwargs['instruction']
        assert "Previous task results available for context:" in instruction
        assert search_result in instruction
        assert "Analyze the retrieved information about USS Monitor" in instruction

    def test_first_task_receives_no_predecessor_results(self, workflow_engine, mock_universal_agent):
        """Test that the first task (no predecessors) receives no predecessor results."""
        # Create single independent task
        tasks = [
            TaskDescription(
                task_name="Independent search task",
                agent_id="search",
                task_type="search",
                prompt="Search for comprehensive information about USS Monitor",
                llm_type="WEAK"
            )
        ]
        
        task_context = TaskContext.from_tasks(tasks, dependencies=[], request_id="independent_test")
        
        # Get the search task
        search_task = list(task_context.task_graph.nodes.values())[0]
        
        # Mock the universal agent
        mock_universal_agent.execute_task.return_value = "Search results found"
        
        # Execute the search task
        workflow_engine.delegate_task(task_context, search_task)
        
        # Verify the universal agent was called
        mock_universal_agent.execute_task.assert_called_once()
        call_args = mock_universal_agent.execute_task.call_args
        
        # The instruction should be the original prompt (no predecessor results)
        instruction = call_args.kwargs['instruction']
        assert instruction == "Search for comprehensive information about USS Monitor"
        assert "Previous task results available for context:" not in instruction

    def test_multiple_predecessor_results_included(self, workflow_engine, mock_universal_agent):
        """Test that tasks with multiple predecessors receive all predecessor results."""
        # Create tasks with multiple dependencies: search + weather -> analysis
        tasks = [
            TaskDescription(
                task_name="Search for location info",
                agent_id="search",
                task_type="search", 
                prompt="Search for location information",
                llm_type="WEAK"
            ),
            TaskDescription(
                task_name="Get weather data",
                agent_id="weather",
                task_type="weather",
                prompt="Get current weather data",
                llm_type="WEAK"
            ),
            TaskDescription(
                task_name="Analyze combined data",
                agent_id="analysis",
                task_type="analysis",
                prompt="Analyze the combined location and weather data",
                llm_type="DEFAULT"
            )
        ]
        
        dependencies = [
            TaskDependency(source="Search for location info", target="Analyze combined data"),
            TaskDependency(source="Get weather data", target="Analyze combined data")
        ]
        
        task_context = TaskContext.from_tasks(tasks, dependencies, request_id="multi_test")
        
        # Complete both predecessor tasks
        search_task = None
        weather_task = None
        analysis_task = None
        
        for task in task_context.task_graph.nodes.values():
            if task.agent_id == "search":
                search_task = task
            elif task.agent_id == "weather":
                weather_task = task
            elif task.agent_id == "analysis":
                analysis_task = task
        
        # Complete predecessor tasks
        search_result = "Location: Seattle, WA - Major city in Pacific Northwest"
        weather_result = "Weather: 65°F, partly cloudy, light rain expected"
        
        task_context.complete_task(search_task.task_id, search_result)
        task_context.complete_task(weather_task.task_id, weather_result)
        
        # Mock the universal agent
        mock_universal_agent.execute_task.return_value = "Combined analysis complete"
        
        # Execute the analysis task
        workflow_engine.delegate_task(task_context, analysis_task)
        
        # Verify the instruction includes both predecessor results
        call_args = mock_universal_agent.execute_task.call_args
        instruction = call_args.kwargs['instruction']
        
        assert "Previous task results available for context:" in instruction
        assert search_result in instruction
        assert weather_result in instruction
        assert "Analyze the combined location and weather data" in instruction

    def test_empty_predecessor_results_handling(self, workflow_engine, mock_universal_agent):
        """Test handling of empty predecessor results."""
        # Create task with predecessor that has empty result
        tasks = [
            TaskDescription(
                task_name="Task with empty result",
                agent_id="search",
                task_type="search",
                prompt="Task that produces empty result",
                llm_type="WEAK"
            ),
            TaskDescription(
                task_name="Dependent task",
                agent_id="analysis",
                task_type="analysis", 
                prompt="Task that depends on previous task",
                llm_type="DEFAULT"
            )
        ]
        
        dependencies = [
            TaskDependency(source="Task with empty result", target="Dependent task")
        ]
        
        task_context = TaskContext.from_tasks(tasks, dependencies, request_id="empty_test")
        
        # Complete first task with empty result
        first_task = list(task_context.task_graph.nodes.values())[0]
        second_task = list(task_context.task_graph.nodes.values())[1]
        
        task_context.complete_task(first_task.task_id, "")  # Empty result
        
        # Mock universal agent
        mock_universal_agent.execute_task.return_value = "Handled empty predecessor"
        
        # Execute dependent task
        workflow_engine.delegate_task(task_context, second_task)
        
        # Verify it doesn't include empty results in enhanced prompt
        call_args = mock_universal_agent.execute_task.call_args
        instruction = call_args.kwargs['instruction']
        
        # Should not include predecessor results section for empty results
        assert instruction == "Task that depends on previous task"
        assert "Previous task results available for context:" not in instruction


class TestTaskResultSharingIntegration:
    """Integration tests for task result sharing with real TaskGraph operations."""
    
    @pytest.fixture
    def mock_llm_factory(self):
        """Create mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        return factory
    
    @pytest.fixture
    def mock_message_bus(self):
        """Create mock message bus."""
        return Mock(spec=MessageBus)
    
    @pytest.fixture
    def mock_universal_agent(self):
        """Create mock universal agent."""
        agent = Mock(spec=UniversalAgent)
        return agent
    
    @pytest.fixture
    def workflow_engine(self, mock_llm_factory, mock_message_bus, mock_universal_agent):
        """Create WorkflowEngine with mocked dependencies."""
        engine = WorkflowEngine(
            llm_factory=mock_llm_factory,
            message_bus=mock_message_bus,
            max_concurrent_tasks=2,
            checkpoint_interval=60
        )
        engine.universal_agent = mock_universal_agent
        return engine
    
    def test_real_task_graph_result_sharing(self):
        """Test result sharing using real TaskGraph operations."""
        # Create real TaskGraph with dependencies
        tasks = [
            TaskDescription(
                task_name="Data collection",
                agent_id="search",
                task_type="search",
                prompt="Collect data about topic",
                llm_type="WEAK"
            ),
            TaskDescription(
                task_name="Data analysis",
                agent_id="analysis",
                task_type="analysis",
                prompt="Analyze collected data",
                llm_type="DEFAULT"
            )
        ]
        
        dependencies = [
            TaskDependency(source="Data collection", target="Data analysis")
        ]
        
        # Create real TaskGraph
        task_graph = TaskGraph(tasks=tasks, dependencies=dependencies, request_id="integration_test")
        
        # Complete first task
        collection_task = None
        analysis_task = None
        
        for task in task_graph.nodes.values():
            if task.agent_id == "search":
                collection_task = task
            elif task.agent_id == "analysis":
                analysis_task = task
        
        # Complete collection task
        collection_result = "Data collected: USS Monitor was a revolutionary ironclad warship"
        task_graph.mark_task_completed(collection_task.task_id, collection_result)
        
        # Get task history for analysis task
        history = task_graph.get_task_history(analysis_task.task_id)
        
        # Verify the history contains the collection result
        assert len(history) == 1
        assert collection_result in history
        
        # Verify the analysis task is now ready
        ready_tasks = task_graph.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == analysis_task.task_id

    def test_task_history_with_include_full_history_flag(self):
        """Test that include_full_history flag works correctly with result sharing."""
        tasks = [
            TaskDescription(
                task_name="Task 1",
                agent_id="agent1",
                task_type="type1",
                prompt="First task",
                include_full_history=False
            ),
            TaskDescription(
                task_name="Task 2", 
                agent_id="agent2",
                task_type="type2",
                prompt="Second task",
                include_full_history=False
            ),
            TaskDescription(
                task_name="Task 3 with full history",
                agent_id="agent3",
                task_type="type3", 
                prompt="Third task needs full history",
                include_full_history=True
            )
        ]
        
        dependencies = [
            TaskDependency(source="Task 1", target="Task 2"),
            TaskDependency(source="Task 2", target="Task 3 with full history")
        ]
        
        task_graph = TaskGraph(tasks=tasks, dependencies=dependencies, request_id="history_test")
        
        # Complete tasks in sequence
        task1 = list(task_graph.nodes.values())[0]
        task2 = list(task_graph.nodes.values())[1] 
        task3 = list(task_graph.nodes.values())[2]
        
        # Complete task 1
        task_graph.mark_task_completed(task1.task_id, "Result from task 1")
        
        # Complete task 2
        task_graph.mark_task_completed(task2.task_id, "Result from task 2")
        
        # Get history for task 3 (should include full history)
        history = task_graph.get_task_history(task3.task_id)
        
        # Task 3 has include_full_history=True, so should get full history
        assert len(history) >= 2  # Should include both previous task results
        
        # Verify it contains results from both previous tasks
        history_str = str(history)
        assert "task 1" in history_str.lower() or "Result from task 1" in history_str
        assert "task 2" in history_str.lower() or "Result from task 2" in history_str

    def test_performance_optimization_verification(self, workflow_engine, mock_universal_agent):
        """Test that result sharing prevents duplicate work in analysis tasks."""
        # Create search -> analysis workflow
        tasks = [
            TaskDescription(
                task_name="Search for information",
                agent_id="search",
                task_type="search",
                prompt="Search for specific information",
                llm_type="WEAK"
            ),
            TaskDescription(
                task_name="Analyze information", 
                agent_id="analysis",
                task_type="analysis",
                prompt="Analyze the information (should not need to search again)",
                llm_type="DEFAULT"
            )
        ]
        
        dependencies = [
            TaskDependency(source="Search for information", target="Analyze information")
        ]
        
        task_context = TaskContext.from_tasks(tasks, dependencies, request_id="performance_test")
        
        # Complete search task
        search_task = list(task_context.task_graph.nodes.values())[0]
        analysis_task = list(task_context.task_graph.nodes.values())[1]
        
        comprehensive_search_result = """Comprehensive search results:
- USS Monitor: Revolutionary ironclad warship
- Designer: John Ericsson
- Battle: Hampton Roads, March 9, 1862
- Fate: Sank off Cape Hatteras, December 1862
- Significance: Changed naval warfare forever"""
        
        task_context.complete_task(search_task.task_id, comprehensive_search_result)
        
        # Mock universal agent to verify it receives the search results
        mock_universal_agent.execute_task.return_value = "Analysis based on provided search results"
        
        # Execute analysis task
        workflow_engine.delegate_task(task_context, analysis_task)
        
        # Verify the analysis task received comprehensive search results
        call_args = mock_universal_agent.execute_task.call_args
        instruction = call_args.kwargs['instruction']
        
        # Should contain all the search result details
        assert "USS Monitor: Revolutionary ironclad warship" in instruction
        assert "John Ericsson" in instruction
        assert "Hampton Roads" in instruction
        assert "Cape Hatteras" in instruction
        assert "Changed naval warfare forever" in instruction
        
        # Should indicate these are previous results
        assert "Previous task results available for context:" in instruction


class TestTaskGraphHistoryMechanism:
    """Test the underlying TaskGraph history mechanism that enables result sharing."""
    
    def test_task_graph_history_storage(self):
        """Test that TaskGraph properly stores task results in history."""
        tasks = [
            TaskDescription(
                task_name="First task",
                agent_id="agent1",
                task_type="type1",
                prompt="First task prompt"
            )
        ]
        
        task_graph = TaskGraph(tasks=tasks, dependencies=[], request_id="history_storage_test")
        task = list(task_graph.nodes.values())[0]
        
        # Initially, history should be empty
        assert len(task_graph.history) == 0
        
        # Complete the task
        result = "Task completed successfully with important data"
        task_graph.mark_task_completed(task.task_id, result)
        
        # History should now contain the task result
        assert len(task_graph.history) == 1
        history_entry = task_graph.history[0]
        
        assert history_entry["task_id"] == task.task_id
        assert history_entry["agent_id"] == "agent1"
        assert history_entry["result"] == result
        assert history_entry["status"] == TaskStatus.COMPLETED

    def test_get_task_history_for_dependent_tasks(self):
        """Test that get_task_history returns predecessor results correctly."""
        tasks = [
            TaskDescription(
                task_name="Producer task",
                agent_id="producer",
                task_type="production",
                prompt="Produce some data"
            ),
            TaskDescription(
                task_name="Consumer task",
                agent_id="consumer",
                task_type="consumption",
                prompt="Consume the produced data"
            )
        ]
        
        dependencies = [
            TaskDependency(source="Producer task", target="Consumer task")
        ]
        
        task_graph = TaskGraph(tasks=tasks, dependencies=dependencies, request_id="dependency_test")
        
        # Get tasks
        producer_task = None
        consumer_task = None
        
        for task in task_graph.nodes.values():
            if task.agent_id == "producer":
                producer_task = task
            elif task.agent_id == "consumer":
                consumer_task = task
        
        # Initially, consumer should have no history (just "The beginning")
        history = task_graph.get_task_history(consumer_task.task_id)
        assert history == ["The beginning"]
        
        # Complete producer task
        producer_result = "Produced data: Important information for consumer"
        task_graph.mark_task_completed(producer_task.task_id, producer_result)
        
        # Now consumer should have access to producer's result
        history = task_graph.get_task_history(consumer_task.task_id)
        assert len(history) == 1
        assert producer_result in history

    def test_task_result_sharing_with_checkpointing(self):
        """Test that task result sharing works correctly with checkpointing."""
        tasks = [
            TaskDescription(
                task_name="Data producer",
                agent_id="search",
                task_type="search",
                prompt="Produce important data"
            ),
            TaskDescription(
                task_name="Data consumer",
                agent_id="analysis",
                task_type="analysis",
                prompt="Consume and analyze the data"
            )
        ]
        
        dependencies = [
            TaskDependency(source="Data producer", target="Data consumer")
        ]
        
        # Create TaskContext
        task_context = TaskContext.from_tasks(tasks, dependencies, request_id="checkpoint_test")
        
        # Complete producer task
        producer_task = None
        consumer_task = None
        
        for task in task_context.task_graph.nodes.values():
            if task.agent_id == "search":
                producer_task = task
            elif task.agent_id == "analysis":
                consumer_task = task
        
        producer_result = "USS Monitor: First ironclad warship commissioned by US Navy"
        task_context.complete_task(producer_task.task_id, producer_result)
        
        # Create checkpoint
        checkpoint = task_context.create_checkpoint()
        
        # Create new context from checkpoint
        restored_context = TaskContext.from_checkpoint(checkpoint)
        
        # Get consumer task from restored context
        restored_consumer_task = None
        for task in restored_context.task_graph.nodes.values():
            if task.agent_id == "analysis":
                restored_consumer_task = task
                break
        
        # Get task history for consumer task from restored context
        history = restored_context.task_graph.get_task_history(restored_consumer_task.task_id)
        
        # Verify producer results are available after checkpoint restoration
        assert len(history) == 1
        assert producer_result in history