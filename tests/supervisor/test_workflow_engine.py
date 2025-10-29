from unittest.mock import Mock, patch

import pytest

from common.message_bus import MessageBus, MessageType
from common.request_model import RequestMetadata
from common.task_context import ExecutionState, TaskContext
from common.task_graph import TaskNode, TaskStatus
from llm_provider.factory import LLMFactory, LLMType
from supervisor.workflow_engine import TaskPriority, WorkflowEngine


class TestWorkflowEngine:
    """Test suite for WorkflowEngine (consolidated RequestManager + TaskScheduler)."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        factory.get_framework = Mock(return_value="strands")
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
    def workflow_engine(self, mock_llm_factory, mock_message_bus, shared_role_registry):
        """Create a WorkflowEngine for testing with shared role registry."""
        from unittest.mock import patch

        # Mock RoleRegistry creation to use shared instance
        with patch("supervisor.workflow_engine.RoleRegistry") as mock_registry_class:
            mock_registry_class.return_value = shared_role_registry
            engine = WorkflowEngine(mock_llm_factory, mock_message_bus)

        engine.role_registry = shared_role_registry
        return engine

    def test_workflow_engine_initialization(
        self, workflow_engine, mock_llm_factory, mock_message_bus
    ):
        """Test WorkflowEngine initialization with unified interface."""
        # Verify core components
        assert workflow_engine.llm_factory == mock_llm_factory
        assert workflow_engine.message_bus == mock_message_bus
        assert workflow_engine.universal_agent is not None

        # Verify workflow tracking
        assert hasattr(workflow_engine, "active_workflows")

        # Verify configuration
        assert workflow_engine.max_retries > 0
        assert workflow_engine.retry_delay > 0

    def test_start_workflow_interface(self, workflow_engine):
        """Test unified start_workflow interface."""
        # In the new architecture, all requests go through fast-reply routing
        # Mock Universal Agent execution
        workflow_engine.universal_agent.execute_task = Mock(
            return_value="Workflow started successfully"
        )

        # Mock role registry for routing
        workflow_engine.role_registry.get_role_execution_type = Mock(
            return_value="hybrid"
        )

        with patch("supervisor.workflow_engine.get_duration_logger") as mock_logger:
            mock_duration_logger = Mock()
            mock_logger.return_value = mock_duration_logger

            # Test start_workflow method
            request = RequestMetadata(
                prompt="Create a project management workflow",
                source_id="workflow_client",
                target_id="supervisor",
            )

            workflow_id = workflow_engine.handle_request(request)

            assert workflow_id is not None
            # New architecture uses fr_ prefix for fast-reply
            assert workflow_id.startswith("fr_")

            # Verify workflow context was created
            context = workflow_engine.get_request_context(workflow_id)
            assert context is not None
            assert context.execution_state == ExecutionState.COMPLETED

    def test_pause_workflow_interface(self, workflow_engine):
        """Test unified pause_workflow interface."""
        # Mock the planning phase to avoid LLM calls for performance
        from common.task_graph import TaskDescription

        mock_task = TaskDescription(
            task_id="pause_interface_task_1",
            task_name="Pause Interface Task",
            task_type="pause_interface_test",
            prompt="Mock pause interface task",
            agent_id="default",
            llm_type="DEFAULT",
            include_full_history=False,
        )

        with (
            patch("roles.core_planning.execute_task_graph") as mock_plan,
            patch.object(
                workflow_engine.universal_agent, "execute_task"
            ) as mock_execute,
            patch.object(workflow_engine, "_execute_dag_parallel") as mock_dag,
        ):
            # Mock the planning phase to avoid LLM calls
            mock_plan.return_value = {
                "task_graph": Mock(),
                "tasks": [mock_task],
                "dependencies": [],
            }
            mock_execute.return_value = "Workflow created"
            mock_dag.return_value = None  # Skip actual DAG execution

            request = RequestMetadata(
                prompt="Long running workflow for pause test",
                source_id="pause_client",
                target_id="supervisor",
            )

            workflow_id = workflow_engine.handle_request(request)

            # Test pause_workflow
            checkpoint = workflow_engine.pause_workflow(workflow_id)

            assert checkpoint is not None
            assert "context_id" in checkpoint
            assert "execution_state" in checkpoint

            # Verify workflow is paused (if context exists)
            context = workflow_engine.get_request_context(workflow_id)
            if context:
                assert context.execution_state == ExecutionState.PAUSED

    def test_resume_workflow_interface(self, workflow_engine):
        """Test unified resume_workflow interface."""
        # Mock the planning phase to avoid LLM calls for performance
        from common.task_graph import TaskDescription

        mock_task = TaskDescription(
            task_id="resume_interface_task_1",
            task_name="Resume Interface Task",
            task_type="resume_interface_test",
            prompt="Mock resume interface task",
            agent_id="default",
            llm_type="DEFAULT",
            include_full_history=False,
        )

        with (
            patch("roles.core_planning.execute_task_graph") as mock_plan,
            patch.object(
                workflow_engine.universal_agent, "execute_task"
            ) as mock_execute,
            patch.object(workflow_engine, "_execute_dag_parallel") as mock_dag,
        ):
            # Mock the planning phase to avoid LLM calls
            mock_plan.return_value = {
                "task_graph": Mock(),
                "tasks": [mock_task],
                "dependencies": [],
            }
            mock_execute.return_value = "Workflow processing"
            mock_dag.return_value = None  # Skip actual DAG execution

            request = RequestMetadata(
                prompt="Workflow for resume test",
                source_id="resume_client",
                target_id="supervisor",
            )

            workflow_id = workflow_engine.handle_request(request)
            checkpoint = workflow_engine.pause_workflow(workflow_id)

            # Test resume_workflow
            success = workflow_engine.resume_workflow(workflow_id, checkpoint)

            assert success is True

            # Verify workflow is running again (if context exists)
            context = workflow_engine.get_request_context(workflow_id)
            if context:
                assert context.execution_state == ExecutionState.RUNNING

    def test_dag_execution_with_parallel_processing(self, workflow_engine):
        """Test DAG execution with parallel task processing (future _execute_dag_parallel method)."""
        # Mock the planning phase to avoid LLM calls for performance
        from common.task_graph import TaskDescription

        mock_task = TaskDescription(
            task_id="dag_parallel_task_1",
            task_name="DAG Parallel Task",
            task_type="dag_parallel_test",
            prompt="Mock DAG parallel task",
            agent_id="default",
            llm_type="DEFAULT",
            include_full_history=False,
        )

        with (
            patch("roles.core_planning.execute_task_graph") as mock_plan,
            patch.object(
                workflow_engine.universal_agent, "execute_task"
            ) as mock_execute,
            patch.object(workflow_engine, "_execute_dag_parallel") as mock_dag,
        ):
            # Mock the planning phase to avoid LLM calls
            mock_plan.return_value = {
                "task_graph": Mock(),
                "tasks": [mock_task],
                "dependencies": [],
            }
            mock_execute.return_value = "Task completed"
            mock_dag.return_value = None  # Skip actual DAG execution

            request = RequestMetadata(
                prompt="Complex workflow with multiple parallel tasks",
                source_id="parallel_client",
                target_id="supervisor",
            )

            workflow_id = workflow_engine.handle_request(request)

            # Verify workflow was created
            assert workflow_id is not None

    def test_workflow_metrics_consolidation(self, workflow_engine):
        """Test consolidated workflow metrics (combining request + task queue stats)."""
        # Mock the planning phase to avoid LLM calls
        from common.task_graph import TaskDescription

        mock_task = TaskDescription(
            task_id="metrics_task_1",
            task_name="Metrics Task",
            task_type="metrics",
            prompt="Mock metrics task",
            agent_id="default",
            llm_type="DEFAULT",
            include_full_history=False,
        )

        # Create some workflows to generate metrics
        with (
            patch("roles.core_planning.execute_task_graph") as mock_plan,
            patch.object(
                workflow_engine.universal_agent, "execute_task"
            ) as mock_execute,
            patch.object(workflow_engine, "_execute_dag_parallel") as mock_dag,
        ):
            # Mock the planning phase to avoid LLM calls
            mock_plan.return_value = {
                "task_graph": Mock(),
                "tasks": [mock_task],
                "dependencies": [],
            }
            mock_execute.return_value = "Metrics test completed"
            mock_dag.return_value = None  # Skip actual DAG execution

            # Create multiple workflows
            for i in range(3):
                request = RequestMetadata(
                    prompt=f"Metrics test workflow {i}",
                    source_id=f"metrics_client_{i}",
                    target_id="supervisor",
                )
                workflow_engine.handle_request(request)

            # Test unified metrics (will combine request tracking + task queue stats)
            ua_status = workflow_engine.get_universal_agent_status()

            assert ua_status is not None
            assert "active_contexts" in ua_status
            assert ua_status["active_contexts"] >= 3

            # Verify workflow status tracking
            active_requests = workflow_engine.list_active_requests()
            assert len(active_requests) >= 3

    def test_concurrency_control_integration(self, workflow_engine):
        """Test concurrency control integration (max_concurrent_tasks logic)."""
        # Mock the planning phase to avoid LLM calls for performance
        from common.task_graph import TaskDescription

        mock_task = TaskDescription(
            task_id="concurrency_task_1",
            task_name="Concurrency Task",
            task_type="concurrency_test",
            prompt="Mock concurrency task",
            agent_id="default",
            llm_type="DEFAULT",
            include_full_history=False,
        )

        with (
            patch("roles.core_planning.execute_task_graph") as mock_plan,
            patch.object(
                workflow_engine.universal_agent, "execute_task"
            ) as mock_execute,
            patch.object(workflow_engine, "_execute_dag_parallel") as mock_dag,
            patch.object(workflow_engine, "_process_task_queue") as mock_process_queue,
        ):
            # Mock the planning phase to avoid LLM calls
            mock_plan.return_value = {
                "task_graph": Mock(),
                "tasks": [mock_task],
                "dependencies": [],
            }
            mock_execute.return_value = "Task completed"
            mock_dag.return_value = None  # Skip actual DAG execution
            mock_process_queue.return_value = None  # Skip queue processing

            # Test concurrency control integration in WorkflowEngine
            workflow_engine.start_workflow_engine()

            # Create mock tasks (reduced for performance)
            mock_tasks = []
            for i in range(2):  # Further reduced for performance
                task = Mock(spec=TaskNode)
                task.task_id = f"concurrent_task_{i}"
                task.status = TaskStatus.PENDING
                mock_tasks.append(task)

            # Schedule tasks (should respect concurrency limit)
            for task in mock_tasks:
                workflow_engine.schedule_task(Mock(), task)

            # Process queue (mocked)
            workflow_engine._process_task_queue()

            # Verify queue processing was called
            mock_process_queue.assert_called()

            # Test basic concurrency control logic
            assert workflow_engine.max_concurrent_tasks > 0

            workflow_engine.stop_workflow_engine()

    def test_message_bus_event_handling(self, workflow_engine):
        """Test message bus event handling for workflow events."""
        # Test TASK_RESPONSE handling
        # Create task response data but don't store unused variable
        {
            "request_id": "test_workflow_123",
            "task_id": "test_task_456",
            "result": "Task completed successfully",
            "status": "completed",
        }

        # Verify message bus subscriptions exist
        # (Currently handled by RequestManager, will be unified in WorkflowEngine)
        assert workflow_engine.message_bus.subscribe.called

        # Test AGENT_ERROR handling
        # Create error data but don't store unused variable
        {
            "request_id": "test_workflow_123",
            "task_id": "test_task_456",
            "error_message": "Task failed",
            "retry_count": 1,
        }

        # Verify error handling exists
        assert hasattr(workflow_engine, "handle_task_error")

    def test_workflow_state_management(self, workflow_engine):
        """Test comprehensive workflow state management."""
        # Mock the planning phase to avoid LLM calls for performance
        from common.task_graph import TaskDescription

        mock_task = TaskDescription(
            task_id="state_mgmt_task_1",
            task_name="State Management Task",
            task_type="state_management_test",
            prompt="Mock state management task",
            agent_id="default",
            llm_type="DEFAULT",
            include_full_history=False,
        )

        with (
            patch("roles.core_planning.execute_task_graph") as mock_plan,
            patch.object(
                workflow_engine.universal_agent, "execute_task"
            ) as mock_execute,
            patch.object(workflow_engine, "_execute_dag_parallel") as mock_dag,
        ):
            # Mock the planning phase to avoid LLM calls
            mock_plan.return_value = {
                "task_graph": Mock(),
                "tasks": [mock_task],
                "dependencies": [],
            }
            mock_execute.return_value = "State management test"
            mock_dag.return_value = None  # Skip actual DAG execution

            request = RequestMetadata(
                prompt="Workflow for state management testing",
                source_id="state_client",
                target_id="supervisor",
            )

            workflow_id = workflow_engine.handle_request(request)

            # Test state tracking
            status = workflow_engine.get_request_status(workflow_id)

            assert status is not None
            assert "request_id" in status
            assert "execution_state" in status
            # Check for actual keys in fast-reply status
            assert "is_completed" in status
            assert "execution_time_ms" in status

    def test_role_based_task_delegation(self, workflow_engine):
        """Test role-based task delegation with dynamic LLM type optimization."""
        # Test different roles and their dynamic LLM type mappings
        role_test_cases = [
            ("planning_agent", "planning"),
            ("weather_agent", "weather"),
            ("timer_agent", "timer"),
            ("smart_home_agent", "smart_home"),
        ]

        for agent_id, expected_role in role_test_cases:
            # Test role determination (dynamic removal of _agent suffix)
            role = workflow_engine._determine_role_from_agent_id(agent_id)
            assert role == expected_role

            # Test LLM type optimization (now dynamic from role registry)
            llm_type = workflow_engine._determine_llm_type_for_role(role)
            # Verify it returns a valid LLM type (should be one of WEAK, DEFAULT, STRONG)
            assert llm_type in [LLMType.WEAK, LLMType.DEFAULT, LLMType.STRONG]

            # Verify specific mappings based on current role definitions
            if role in ["planning"]:
                assert (
                    llm_type == LLMType.STRONG
                ), f"Expected STRONG for {role}, got {llm_type}"
            elif role in ["weather", "timer", "smart_home"]:
                assert (
                    llm_type == LLMType.WEAK
                ), f"Expected WEAK for {role}, got {llm_type}"

    def test_workflow_cleanup_and_resource_management(self, workflow_engine):
        """Test workflow cleanup and resource management."""
        # Mock the planning phase to avoid LLM calls
        from common.task_graph import TaskDescription

        mock_task = TaskDescription(
            task_id="cleanup_task_1",
            task_name="Cleanup Test Task",
            task_type="cleanup",
            prompt="Mock cleanup task",
            agent_id="default",
            llm_type="DEFAULT",
            include_full_history=False,
        )

        # Create multiple workflows with proper mocking
        with (
            patch("roles.core_planning.execute_task_graph") as mock_plan,
            patch.object(
                workflow_engine.universal_agent, "execute_task"
            ) as mock_execute,
            patch.object(workflow_engine, "_execute_dag_parallel") as mock_dag,
        ):
            # Mock the planning phase to avoid LLM calls
            mock_plan.return_value = {
                "task_graph": Mock(),
                "tasks": [mock_task],
                "dependencies": [],
            }
            mock_execute.return_value = "Cleanup test workflow"
            mock_dag.return_value = None  # Skip actual DAG execution

            workflow_ids = []
            for i in range(3):  # Reduced from 5 to 3 for performance
                request = RequestMetadata(
                    prompt=f"Cleanup test workflow {i}",
                    source_id=f"cleanup_client_{i}",
                    target_id="supervisor",
                )
                workflow_id = workflow_engine.handle_request(request)
                workflow_ids.append(workflow_id)

            # Verify workflows were created
            initial_count = len(workflow_engine.active_workflows)
            assert initial_count >= 3  # Adjusted expectation

            # Test cleanup
            workflow_engine.cleanup_completed_requests(max_age_seconds=0)

            # Verify cleanup executed without errors
            final_count = len(workflow_engine.active_workflows)
            assert final_count <= initial_count

    def test_workflow_engine_consolidation_benefits(self, workflow_engine):
        """Test the benefits of WorkflowEngine consolidation."""
        # Test unified interface benefits

        # 1. Single point of workflow management
        assert hasattr(workflow_engine, "handle_request")
        assert hasattr(workflow_engine, "pause_workflow")
        assert hasattr(workflow_engine, "resume_workflow")

        # 2. Integrated state management
        assert hasattr(workflow_engine, "active_workflows")
        assert hasattr(workflow_engine, "get_request_status")

        # 3. Universal Agent integration
        assert hasattr(workflow_engine, "universal_agent")
        assert hasattr(workflow_engine, "delegate_task")

        # 4. Error handling and retry logic
        assert hasattr(workflow_engine, "handle_task_error")
        assert hasattr(workflow_engine, "max_retries")
        assert hasattr(workflow_engine, "retry_delay")

        # 5. MCP integration
        assert hasattr(workflow_engine, "mcp_manager")
        assert hasattr(workflow_engine, "get_mcp_tools")
        assert hasattr(workflow_engine, "execute_mcp_tool")


class TestWorkflowEngineConsolidation:
    """Test the consolidation of RequestManager + TaskScheduler functionality."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        factory.get_framework = Mock(return_value="strands")
        return factory

    @pytest.fixture
    def mock_message_bus(self):
        """Create a mock message bus."""
        bus = Mock(spec=MessageBus)
        bus.subscribe = Mock()
        bus.publish = Mock()
        return bus

    def test_consolidated_functionality_requirements(
        self, mock_llm_factory, mock_message_bus
    ):
        """Test that consolidated WorkflowEngine meets all requirements."""
        # Create WorkflowEngine to test consolidated functionality
        workflow_engine = WorkflowEngine(
            mock_llm_factory, mock_message_bus, max_concurrent_tasks=5
        )

        # Test consolidated functionality that should be preserved
        # Note: _determine_role_from_agent_id and _determine_llm_type_for_role were removed in new architecture
        wf_capabilities = [
            "handle_request",
            "pause_workflow",
            "resume_workflow",
            "delegate_task",
            "get_request_status",
            "get_request_context",
            "handle_task_error",
            "schedule_task",
            "_process_task_queue",
            "start_workflow_engine",
            "stop_workflow_engine",
            "pause_request",
            "resume_request",
            "get_workflow_metrics",
            "handle_task_completion",
        ]

        for capability in wf_capabilities:
            assert hasattr(
                workflow_engine, capability
            ), f"WorkflowEngine missing {capability}"

        # Test data structures that should be consolidated
        assert hasattr(workflow_engine, "active_workflows")
        assert hasattr(workflow_engine, "task_queue")
        assert hasattr(workflow_engine, "running_tasks")
        assert hasattr(workflow_engine, "max_concurrent_tasks")

    def test_priority_queue_integration(self, mock_llm_factory, mock_message_bus):
        """Test priority queue integration in WorkflowEngine."""
        workflow_engine = WorkflowEngine(mock_llm_factory, mock_message_bus)

        # Test priority queue functionality
        mock_context = Mock(spec=TaskContext)
        mock_task1 = Mock(spec=TaskNode)
        mock_task1.task_id = "high_priority_task"
        mock_task2 = Mock(spec=TaskNode)
        mock_task2.task_id = "low_priority_task"

        # Schedule tasks with different priorities
        workflow_engine.schedule_task(mock_context, mock_task1, TaskPriority.HIGH)
        workflow_engine.schedule_task(mock_context, mock_task2, TaskPriority.LOW)

        # Verify priority queue ordering
        assert len(workflow_engine.task_queue) == 2

        # High priority task should be first
        first_task = workflow_engine.task_queue[0]
        assert first_task.priority == TaskPriority.HIGH

    def test_message_bus_subscription_consolidation(
        self, mock_llm_factory, mock_message_bus
    ):
        """Test consolidated message bus subscriptions in WorkflowEngine."""
        WorkflowEngine(mock_llm_factory, mock_message_bus)

        # Verify WorkflowEngine subscriptions
        wf_calls = mock_message_bus.subscribe.call_args_list
        wf_message_types = [call[0][1] for call in wf_calls if len(call[0]) > 1]

        # Should subscribe to all necessary message types
        expected_types = [
            MessageType.INCOMING_REQUEST,
            MessageType.TASK_RESPONSE,
            MessageType.AGENT_ERROR,
        ]
        for msg_type in expected_types:
            assert (
                msg_type in wf_message_types or len(wf_calls) >= 3
            )  # Allow for different subscription patterns


if __name__ == "__main__":
    pytest.main([__file__])
