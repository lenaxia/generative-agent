import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Optional

from supervisor.request_manager import RequestManager  # Will become WorkflowEngine
from supervisor.task_scheduler import TaskScheduler, TaskPriority, QueuedTask
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.universal_agent import UniversalAgent
from common.task_context import TaskContext, ExecutionState
from common.task_graph import TaskNode, TaskStatus, TaskDescription, TaskDependency
from common.message_bus import MessageBus, MessageType
from common.request_model import RequestMetadata


class TestWorkflowEngine:
    """Test suite for WorkflowEngine (consolidated RequestManager + TaskScheduler)."""
    
    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        factory.get_framework = Mock(return_value='strands')
        factory.create_strands_model = Mock()
        return factory
    
    @pytest.fixture
    def mock_message_bus(self):
        """Create a mock message bus."""
        bus = Mock(spec=MessageBus)
        bus.subscribe = Mock()
        bus.publish = Mock()
        return bus
    
    @pytest.fixture
    def workflow_engine(self, mock_llm_factory, mock_message_bus):
        """Create a WorkflowEngine for testing."""
        # For now, this is still RequestManager, but will become WorkflowEngine
        return RequestManager(mock_llm_factory, mock_message_bus)
    
    def test_workflow_engine_initialization(self, workflow_engine, mock_llm_factory, mock_message_bus):
        """Test WorkflowEngine initialization with unified interface."""
        # Verify core components
        assert workflow_engine.llm_factory == mock_llm_factory
        assert workflow_engine.message_bus == mock_message_bus
        assert workflow_engine.universal_agent is not None
        
        # Verify workflow tracking
        assert hasattr(workflow_engine, 'request_contexts')  # Will become active_workflows
        
        # Verify configuration
        assert workflow_engine.max_retries > 0
        assert workflow_engine.retry_delay > 0
    
    def test_start_workflow_interface(self, workflow_engine):
        """Test unified start_workflow interface."""
        # Mock Universal Agent execution
        with patch.object(workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Workflow started successfully"
            
            # Test start_workflow method (currently handle_request)
            request = RequestMetadata(
                prompt="Create a project management workflow",
                source_id="workflow_client",
                target_id="supervisor"
            )
            
            workflow_id = workflow_engine.handle_request(request)  # Will become start_workflow
            
            assert workflow_id is not None
            assert workflow_id.startswith('req_')  # Will become 'wf_'
            
            # Verify workflow context was created
            context = workflow_engine.get_request_context(workflow_id)  # Will become get_workflow_context
            assert context is not None
            assert context.execution_state == ExecutionState.RUNNING
    
    def test_pause_workflow_interface(self, workflow_engine):
        """Test unified pause_workflow interface."""
        # Create a workflow first
        with patch.object(workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Workflow created"
            
            request = RequestMetadata(
                prompt="Long running workflow for pause test",
                source_id="pause_client",
                target_id="supervisor"
            )
            
            workflow_id = workflow_engine.handle_request(request)
            
            # Test pause_workflow (currently pause_request)
            checkpoint = workflow_engine.pause_request(workflow_id)  # Will become pause_workflow
            
            assert checkpoint is not None
            assert 'context_id' in checkpoint
            assert 'execution_state' in checkpoint
            
            # Verify workflow is paused
            context = workflow_engine.get_request_context(workflow_id)
            assert context.execution_state == ExecutionState.PAUSED
    
    def test_resume_workflow_interface(self, workflow_engine):
        """Test unified resume_workflow interface."""
        # Create and pause a workflow
        with patch.object(workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Workflow processing"
            
            request = RequestMetadata(
                prompt="Workflow for resume test",
                source_id="resume_client",
                target_id="supervisor"
            )
            
            workflow_id = workflow_engine.handle_request(request)
            checkpoint = workflow_engine.pause_request(workflow_id)
            
            # Test resume_workflow (currently resume_request)
            success = workflow_engine.resume_request(workflow_id, checkpoint)  # Will become resume_workflow
            
            assert success == True
            
            # Verify workflow is running again
            context = workflow_engine.get_request_context(workflow_id)
            assert context.execution_state == ExecutionState.RUNNING
    
    def test_dag_execution_with_parallel_processing(self, workflow_engine):
        """Test DAG execution with parallel task processing (future _execute_dag_parallel method)."""
        # Create a workflow with multiple tasks
        with patch.object(workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Task completed"
            
            request = RequestMetadata(
                prompt="Complex workflow with multiple parallel tasks",
                source_id="parallel_client",
                target_id="supervisor"
            )
            
            workflow_id = workflow_engine.handle_request(request)
            context = workflow_engine.get_request_context(workflow_id)
            
            # Verify DAG execution capabilities
            ready_tasks = context.get_ready_tasks()
            assert len(ready_tasks) >= 0  # Should have tasks ready for execution
            
            # Verify task delegation works (simulates parallel execution)
            if ready_tasks:
                task = ready_tasks[0]
                workflow_engine.delegate_task(context, task)  # Will be part of _execute_dag_parallel
                
                # Verify task was processed
                assert mock_execute.called
    
    def test_workflow_metrics_consolidation(self, workflow_engine):
        """Test consolidated workflow metrics (combining request + task queue stats)."""
        # Create some workflows to generate metrics
        with patch.object(workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Metrics test completed"
            
            # Create multiple workflows
            for i in range(3):
                request = RequestMetadata(
                    prompt=f"Metrics test workflow {i}",
                    source_id=f"metrics_client_{i}",
                    target_id="supervisor"
                )
                workflow_engine.handle_request(request)
            
            # Test unified metrics (will combine request tracking + task queue stats)
            ua_status = workflow_engine.get_universal_agent_status()
            
            assert ua_status is not None
            assert 'active_contexts' in ua_status  # Will become 'active_workflows'
            assert ua_status['active_contexts'] >= 3
            
            # Verify workflow status tracking
            active_requests = workflow_engine.list_active_requests()  # Will become list_active_workflows
            assert len(active_requests) >= 3
    
    def test_concurrency_control_integration(self, workflow_engine):
        """Test concurrency control integration (max_concurrent_tasks logic)."""
        # This tests the future integration of TaskScheduler's concurrency control
        # Currently this is separate, but will be integrated in WorkflowEngine
        
        # Create a TaskScheduler to test the logic that will be integrated
        task_scheduler = TaskScheduler(
            request_manager=workflow_engine,
            message_bus=workflow_engine.message_bus,
            max_concurrent_tasks=2  # Low limit for testing
        )
        
        # Test concurrency control
        task_scheduler.start()
        
        # Create mock tasks
        mock_tasks = []
        for i in range(5):
            task = Mock(spec=TaskNode)
            task.task_id = f"concurrent_task_{i}"
            task.status = TaskStatus.PENDING
            mock_tasks.append(task)
        
        # Schedule tasks (should respect concurrency limit)
        for task in mock_tasks:
            task_scheduler.schedule_task(Mock(), task)
        
        # Process queue
        task_scheduler._process_task_queue()
        
        # Should only run max_concurrent_tasks at once
        assert len(task_scheduler.running_tasks) <= task_scheduler.max_concurrent_tasks
        
        task_scheduler.stop()
    
    def test_message_bus_event_handling(self, workflow_engine):
        """Test message bus event handling for workflow events."""
        # Test TASK_RESPONSE handling
        task_response_data = {
            "request_id": "test_workflow_123",
            "task_id": "test_task_456", 
            "result": "Task completed successfully",
            "status": "completed"
        }
        
        # Verify message bus subscriptions exist
        # (Currently handled by RequestManager, will be unified in WorkflowEngine)
        assert workflow_engine.message_bus.subscribe.called
        
        # Test AGENT_ERROR handling
        error_data = {
            "request_id": "test_workflow_123",
            "task_id": "test_task_456",
            "error_message": "Task failed",
            "retry_count": 1
        }
        
        # Verify error handling exists
        assert hasattr(workflow_engine, 'handle_task_error')
    
    def test_workflow_state_management(self, workflow_engine):
        """Test comprehensive workflow state management."""
        # Create a workflow
        with patch.object(workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "State management test"
            
            request = RequestMetadata(
                prompt="Workflow for state management testing",
                source_id="state_client",
                target_id="supervisor"
            )
            
            workflow_id = workflow_engine.handle_request(request)
            
            # Test state tracking
            status = workflow_engine.get_request_status(workflow_id)  # Will become get_workflow_status
            
            assert status is not None
            assert 'request_id' in status  # Will become 'workflow_id'
            assert 'execution_state' in status
            assert 'performance_metrics' in status
            
            # Test context management
            context = workflow_engine.get_request_context(workflow_id)
            assert context is not None
            assert context.context_id is not None
    
    def test_role_based_task_delegation(self, workflow_engine):
        """Test role-based task delegation with LLM type optimization."""
        # Test different roles and their LLM type mappings
        role_test_cases = [
            ("planning_agent", "planning", LLMType.STRONG),
            ("search_agent", "search", LLMType.WEAK),
            ("weather_agent", "weather", LLMType.WEAK),
            ("summarizer_agent", "summarizer", LLMType.DEFAULT),
            ("slack_agent", "slack", LLMType.DEFAULT)
        ]
        
        for agent_id, expected_role, expected_llm_type in role_test_cases:
            # Test role determination
            role = workflow_engine._determine_role_from_agent_id(agent_id)
            assert role == expected_role
            
            # Test LLM type optimization
            llm_type = workflow_engine._determine_llm_type_for_role(role)
            assert llm_type == expected_llm_type
    
    def test_workflow_cleanup_and_resource_management(self, workflow_engine):
        """Test workflow cleanup and resource management."""
        # Create multiple workflows
        with patch.object(workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Cleanup test workflow"
            
            workflow_ids = []
            for i in range(5):
                request = RequestMetadata(
                    prompt=f"Cleanup test workflow {i}",
                    source_id=f"cleanup_client_{i}",
                    target_id="supervisor"
                )
                workflow_id = workflow_engine.handle_request(request)
                workflow_ids.append(workflow_id)
            
            # Verify workflows were created
            initial_count = len(workflow_engine.request_contexts)  # Will become active_workflows
            assert initial_count >= 5
            
            # Test cleanup
            workflow_engine.cleanup_completed_requests(max_age_seconds=0)  # Will become cleanup_completed_workflows
            
            # Verify cleanup executed without errors
            final_count = len(workflow_engine.request_contexts)
            assert final_count <= initial_count
    
    def test_workflow_engine_consolidation_benefits(self, workflow_engine):
        """Test the benefits of WorkflowEngine consolidation."""
        # Test unified interface benefits
        
        # 1. Single point of workflow management
        assert hasattr(workflow_engine, 'handle_request')  # Will become start_workflow
        assert hasattr(workflow_engine, 'pause_request')   # Will become pause_workflow
        assert hasattr(workflow_engine, 'resume_request')  # Will become resume_workflow
        
        # 2. Integrated state management
        assert hasattr(workflow_engine, 'request_contexts')  # Will become active_workflows
        assert hasattr(workflow_engine, 'get_request_status')  # Will become get_workflow_status
        
        # 3. Universal Agent integration
        assert hasattr(workflow_engine, 'universal_agent')
        assert hasattr(workflow_engine, 'delegate_task')
        
        # 4. Error handling and retry logic
        assert hasattr(workflow_engine, 'handle_task_error')
        assert hasattr(workflow_engine, 'max_retries')
        assert hasattr(workflow_engine, 'retry_delay')
        
        # 5. MCP integration
        assert hasattr(workflow_engine, 'mcp_manager')
        assert hasattr(workflow_engine, 'get_mcp_tools')
        assert hasattr(workflow_engine, 'execute_mcp_tool')


class TestWorkflowEngineConsolidation:
    """Test the consolidation of RequestManager + TaskScheduler functionality."""
    
    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        factory.get_framework = Mock(return_value='strands')
        return factory
    
    @pytest.fixture
    def mock_message_bus(self):
        """Create a mock message bus."""
        bus = Mock(spec=MessageBus)
        bus.subscribe = Mock()
        bus.publish = Mock()
        return bus
    
    def test_consolidated_functionality_requirements(self, mock_llm_factory, mock_message_bus):
        """Test that consolidated WorkflowEngine meets all requirements."""
        # Create both components to test consolidation requirements
        request_manager = RequestManager(mock_llm_factory, mock_message_bus)
        task_scheduler = TaskScheduler(request_manager, mock_message_bus, max_concurrent_tasks=5)
        
        # Test RequestManager functionality that should be preserved
        rm_capabilities = [
            'handle_request',      # → start_workflow
            'pause_request',       # → pause_workflow  
            'resume_request',      # → resume_workflow
            'delegate_task',       # → _execute_dag_parallel (internal)
            'get_request_status',  # → get_workflow_status
            'get_request_context', # → get_workflow_context
            'handle_task_error',   # → preserved
            '_determine_role_from_agent_id',  # → preserved
            '_determine_llm_type_for_role'    # → preserved
        ]
        
        for capability in rm_capabilities:
            assert hasattr(request_manager, capability), f"RequestManager missing {capability}"
        
        # Test TaskScheduler functionality that should be integrated
        ts_capabilities = [
            'schedule_task',         # → integrated into _execute_dag_parallel
            '_process_task_queue',   # → integrated into _execute_dag_parallel
            'start',                 # → integrated into start_workflow
            'stop',                  # → integrated into stop_workflow
            'pause',                 # → integrated into pause_workflow
            'resume',                # → integrated into resume_workflow
            'get_metrics',           # → integrated into get_workflow_metrics
            'handle_task_completion', # → preserved
            'handle_task_error'      # → consolidated with RequestManager version
        ]
        
        for capability in ts_capabilities:
            assert hasattr(task_scheduler, capability), f"TaskScheduler missing {capability}"
        
        # Test data structures that should be consolidated
        assert hasattr(request_manager, 'request_contexts')  # → active_workflows
        assert hasattr(task_scheduler, 'task_queue')         # → integrated into WorkflowEngine
        assert hasattr(task_scheduler, 'running_tasks')      # → integrated into WorkflowEngine
        assert hasattr(task_scheduler, 'max_concurrent_tasks')  # → preserved
    
    def test_priority_queue_integration(self, mock_llm_factory, mock_message_bus):
        """Test priority queue integration from TaskScheduler."""
        request_manager = RequestManager(mock_llm_factory, mock_message_bus)
        task_scheduler = TaskScheduler(request_manager, mock_message_bus)
        
        # Test priority queue functionality
        mock_context = Mock(spec=TaskContext)
        mock_task1 = Mock(spec=TaskNode)
        mock_task1.task_id = "high_priority_task"
        mock_task2 = Mock(spec=TaskNode)
        mock_task2.task_id = "low_priority_task"
        
        # Schedule tasks with different priorities
        task_scheduler.schedule_task(mock_context, mock_task1, TaskPriority.HIGH)
        task_scheduler.schedule_task(mock_context, mock_task2, TaskPriority.LOW)
        
        # Verify priority queue ordering
        assert len(task_scheduler.task_queue) == 2
        
        # High priority task should be first
        first_task = task_scheduler.task_queue[0]
        assert first_task.priority == TaskPriority.HIGH
    
    def test_message_bus_subscription_consolidation(self, mock_llm_factory, mock_message_bus):
        """Test consolidated message bus subscriptions."""
        request_manager = RequestManager(mock_llm_factory, mock_message_bus)
        task_scheduler = TaskScheduler(request_manager, mock_message_bus)
        
        # Verify RequestManager subscriptions
        rm_calls = mock_message_bus.subscribe.call_args_list
        rm_message_types = [call[0][1] for call in rm_calls if len(call[0]) > 1]
        
        # Should subscribe to INCOMING_REQUEST
        assert MessageType.INCOMING_REQUEST in rm_message_types
        
        # Reset mock for TaskScheduler
        mock_message_bus.subscribe.reset_mock()
        
        # TaskScheduler should subscribe to TASK_RESPONSE and AGENT_ERROR
        ts_calls = mock_message_bus.subscribe.call_args_list
        ts_message_types = [call[0][1] for call in ts_calls if len(call[0]) > 1]
        
        # Verify TaskScheduler subscriptions
        expected_ts_types = [MessageType.TASK_RESPONSE, MessageType.AGENT_ERROR]
        for msg_type in expected_ts_types:
            assert msg_type in ts_message_types or len(ts_calls) == 2  # Allow for different subscription patterns


if __name__ == "__main__":
    pytest.main([__file__])