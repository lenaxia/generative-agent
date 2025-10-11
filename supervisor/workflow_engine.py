"""Workflow engine module for managing and executing complex workflows.

This module provides the WorkflowEngine class that handles workflow execution,
task scheduling, dependency management, and coordination of complex multi-step
processes within the supervisor system.
"""

import heapq
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import yaml

from common.message_bus import MessageBus, MessageType
from common.request_model import RequestMetadata
from common.task_context import TaskContext
from common.task_graph import (
    TaskDependency,
    TaskDescription,
    TaskGraph,
    TaskNode,
    TaskStatus,
)
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.mcp_client import MCPClientManager
from llm_provider.request_router import FastPathRoutingConfig, RequestRouter
from llm_provider.role_registry import RoleRegistry
from llm_provider.universal_agent import UniversalAgent
from supervisor.workflow_duration_logger import (
    WorkflowSource,
    WorkflowType,
    get_duration_logger,
)

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels for scheduling."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class WorkflowState(str, Enum):
    """Workflow execution states."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


@dataclass
class QueuedTask:
    """Represents a task in the workflow queue."""

    priority: TaskPriority
    scheduled_time: float
    task: TaskNode
    context: TaskContext
    task_id: str = field(init=False)

    def __post_init__(self):
        """Post-initialization processing for ScheduledTask.

        Automatically extracts and sets the task_id from the associated
        TaskNode for consistent task identification in the workflow engine.
        """
        self.task_id = self.task.task_id

    def __lt__(self, other):
        """Priority queue ordering: higher priority first, then by scheduled time."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value  # Higher priority first
        return (
            self.scheduled_time < other.scheduled_time
        )  # Earlier scheduled time first


class WorkflowEngine:
    """Unified workflow management with DAG execution and state persistence.

    This consolidates RequestManager + TaskScheduler functionality into a single
    WorkflowEngine for simplified architecture and better performance.

    Combines:
    - Request lifecycle management (from RequestManager)
    - Task scheduling and priority queuing (from TaskScheduler)
    - DAG execution with concurrency control
    - Universal Agent integration with StrandsAgent framework
    """

    def __init__(
        self,
        llm_factory: LLMFactory,
        message_bus: MessageBus,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrent_tasks: int = 5,
        checkpoint_interval: int = 300,
        mcp_config_path: Optional[str] = None,
        roles_directory: str = "roles",
        fast_path_config: Optional[dict[str, Any]] = None,
    ):
        """Initialize WorkflowEngine with Universal Agent, task scheduling, and fast-path routing support.

        Args:
            llm_factory: LLMFactory for creating Universal Agent
            message_bus: Message bus for task distribution
            max_retries: Maximum retry attempts for failed tasks
            retry_delay: Delay between retry attempts
            max_concurrent_tasks: Maximum concurrent task execution
            checkpoint_interval: Checkpoint creation interval in seconds
            mcp_config_path: Optional path to MCP configuration file
            roles_directory: Directory containing role definitions
            fast_path_config: Optional fast-path routing configuration
        """
        self.llm_factory = llm_factory
        self.message_bus = message_bus

        # Initialize MCP manager
        self.mcp_manager = self._initialize_mcp_manager(mcp_config_path)

        # Initialize role registry with MessageBus for dynamic event registration
        self.role_registry = RoleRegistry(roles_directory, message_bus=self.message_bus)

        # Inject dependencies into MessageBus for event handler wrapping
        self.message_bus.workflow_engine = self
        self.message_bus.llm_factory = self.llm_factory
        # communication_manager will be injected by Supervisor after WorkflowEngine initialization

        # Set this as the global registry to ensure consistency across the system
        RoleRegistry._global_registry = self.role_registry
        # Use initialize_once for performance - avoid repeated loading
        self.role_registry.initialize_once()

        # Get fast-reply roles count for logging (uses cache)
        fast_reply_roles = self.role_registry.get_fast_reply_roles()
        logger.info(
            f"Initialized with {len(fast_reply_roles)} fast-reply roles: {[r.name for r in fast_reply_roles]}"
        )

        # Create Universal Agent with MCP and role support first
        self.universal_agent = UniversalAgent(
            llm_factory, role_registry=self.role_registry, mcp_manager=self.mcp_manager
        )

        # Performance optimization: Warm up models and agents for faster routing
        logger.info("Warming up LLM models for optimal performance...")
        try:
            llm_factory.warm_models()
            logger.info("Model warming completed successfully")
        except Exception as e:
            logger.warning(f"Model warming failed (non-critical): {e}")

        # Performance optimization: Pre-warm agent pool to avoid MetricsClient creation during workflows
        logger.info("Pre-warming agent pool to eliminate workflow latency...")
        try:
            llm_factory.warm_agent_pool()
            logger.info("Agent pool warming completed successfully")
        except Exception as e:
            logger.warning(f"Agent pool warming failed (non-critical): {e}")

        # Initialize fast-path routing with UniversalAgent
        self.fast_path_config = FastPathRoutingConfig.from_dict(fast_path_config or {})
        self.request_router = RequestRouter(
            llm_factory, self.role_registry, self.universal_agent
        )

        # Workflow tracking
        self.active_workflows: dict[str, TaskContext] = {}

        # Task scheduling
        self.task_queue: list[QueuedTask] = []
        self.running_tasks: dict[str, dict] = {}
        self.max_concurrent_tasks = max_concurrent_tasks
        self.checkpoint_interval = checkpoint_interval

        # Workflow state
        self.state = WorkflowState.IDLE
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.last_checkpoint_time: Optional[float] = None

        # Configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Subscribe to workflow events
        self.message_bus.subscribe(
            self, MessageType.INCOMING_REQUEST, self.handle_request
        )
        self.message_bus.subscribe(
            self, MessageType.TASK_RESPONSE, self.handle_task_completion
        )
        self.message_bus.subscribe(
            self, MessageType.AGENT_ERROR, self.handle_task_error_event
        )

        logger.info(
            "WorkflowEngine initialized with Universal Agent, task scheduling, and MCP integration"
        )

    # ==================== UNIFIED WORKFLOW INTERFACE ====================

    def start_workflow(
        self, instruction: str, source_id: str = "client", target_id: str = "supervisor"
    ) -> str:
        """Create and start a new workflow from user instruction.

        Args:
            instruction: User instruction to create workflow for
            source_id: Source identifier for the workflow
            target_id: Target identifier for the workflow

        Returns:
            str: Workflow ID for tracking
        """
        request = RequestMetadata(
            prompt=instruction, source_id=source_id, target_id=target_id
        )
        workflow_id = self.handle_request(request)

        # Update workflow source to CLI for workflows started via start_workflow
        if workflow_id and hasattr(self, "duration_logger"):
            try:
                if workflow_id in self.duration_logger.active_workflows:
                    self.duration_logger.active_workflows[
                        workflow_id
                    ].source = WorkflowSource.CLI
            except Exception as e:
                logger.debug(f"Could not update workflow source for {workflow_id}: {e}")

        return workflow_id

    def handle_request(self, request: RequestMetadata) -> str:
        """Enhanced request handling with fast-path routing.

        Args:
            request: The incoming request metadata

        Returns:
            str: Request ID for tracking
        """
        # Fast-path routing (if enabled)
        if self.fast_path_config.enabled:
            routing_result = self.request_router.route_request(request.prompt)

            if (
                routing_result["route"] != "PLANNING"
                and routing_result.get("confidence", 0)
                >= self.fast_path_config.confidence_threshold
            ):
                # Fast-path execution
                return self._handle_fast_reply(request, routing_result)

        # Existing complex workflow path (unchanged)
        return self._handle_complex_workflow(request)

    def _handle_fast_reply(self, request: RequestMetadata, routing_result: dict) -> str:
        """Execute fast-reply with unified TaskContext storage."""
        try:
            request_id = "fr_" + str(uuid.uuid4()).split("-")[-1]
            role = routing_result["route"]
            parameters = routing_result.get("parameters", {})

            logger.info(
                f"Fast-reply '{request_id}' via {role} role with {len(parameters)} parameters"
            )

            # Start duration tracking
            duration_logger = get_duration_logger()
            duration_logger.start_workflow_tracking(
                workflow_id=request_id,
                source=WorkflowSource.CLI,
                workflow_type=WorkflowType.FAST_REPLY,
                instruction=request.prompt,
            )

            # Check if this is a hybrid role
            execution_type = self.role_registry.get_role_execution_type(role)

            if execution_type == "hybrid":
                # Get workflow context for user_id and channel_id
                user_id = None
                channel_id = None

                # First try to get from request metadata
                if request.metadata:
                    user_id = request.metadata.get("user_id")
                    channel_id = request.metadata.get("channel_id")
                    logger.info(
                        f"Fast-reply context from request metadata: user_id={user_id}, channel_id={channel_id}"
                    )

                # Fallback to workflow metrics if not in metadata
                if not user_id or not channel_id:
                    if (
                        hasattr(self, "duration_logger")
                        and request_id in self.duration_logger.active_workflows
                    ):
                        workflow_metrics = self.duration_logger.active_workflows[
                            request_id
                        ]
                        if (
                            hasattr(workflow_metrics, "user_id")
                            and workflow_metrics.user_id
                        ):
                            user_id = workflow_metrics.user_id
                        if (
                            hasattr(workflow_metrics, "channel_id")
                            and workflow_metrics.channel_id
                        ):
                            channel_id = workflow_metrics.channel_id

                # Create TaskContext with user_id and channel_id for lifecycle functions
                from common.task_context import TaskContext
                from common.task_graph import TaskGraph

                task_graph = TaskGraph(tasks=[], dependencies=[])
                task_context = TaskContext(
                    task_graph=task_graph,
                    context_id=request_id,
                    user_id=user_id,
                    channel_id=channel_id,
                )

                # Execute hybrid role with pre-extracted parameters
                # execute_task is now synchronous and handles async internally
                result = self.universal_agent.execute_task(
                    instruction=request.prompt,
                    role=role,
                    context=task_context,
                    extracted_parameters=parameters,
                )
            else:
                # Existing LLM execution path with parameter context injection
                if parameters:
                    param_context = "Context: " + ", ".join(
                        [f"{k}={v}" for k, v in parameters.items()]
                    )
                    enhanced_instruction = f"{param_context}\n\n{request.prompt}"
                else:
                    enhanced_instruction = request.prompt

                result = self.universal_agent.execute_task(
                    instruction=enhanced_instruction,
                    role=role,
                    llm_type=LLMType.WEAK,
                    context=None,
                )

            # Complete duration tracking
            duration_logger.complete_workflow_tracking(
                workflow_id=request_id,
                success=True,
                role=role,
                confidence=routing_result.get("confidence"),
            )

            # NEW: Create minimal TaskContext with completed TaskNode for unified storage
            task_node = TaskNode(
                task_id=request_id,
                task_name=f"fast_reply_{role}",
                request_id=request_id,
                agent_id=role,
                task_type="fast_reply",
                prompt=request.prompt,
                status=TaskStatus.COMPLETED,
                result=result,
                role=role,
                llm_type="WEAK",
                start_time=time.time(),
                duration=0.0,  # Fast replies are essentially instantaneous
                task_context={
                    "confidence": routing_result.get("confidence"),
                    "parameters": parameters,
                    "execution_type": execution_type,
                },
            )

            # Create TaskGraph with single completed node
            task_graph = TaskGraph(tasks=[], dependencies=[], request_id=request_id)
            task_graph.nodes[request_id] = task_node

            # Create TaskContext and store in active_workflows for unified retrieval
            from common.task_context import ExecutionState, TaskContext

            task_context = TaskContext(task_graph, context_id=request_id)
            task_context.execution_state = ExecutionState.COMPLETED
            task_context.end_time = time.time()
            self.active_workflows[request_id] = task_context

            logger.info(f"Fast-reply '{request_id}' stored in unified TaskContext")

            # Send response back to the requester if response_requested is True
            if request.response_requested:
                logger.info(
                    f"📤 Sending fast-reply result back to requester: {request.metadata.get('channel_id')}"
                )
                self.message_bus.publish(
                    self,
                    MessageType.SEND_MESSAGE,
                    {
                        "message": result,
                        "context": {
                            "channel_id": request.metadata.get("channel_id"),
                            "user_id": request.metadata.get("user_id"),
                            "request_id": request.metadata.get("request_id"),
                        },
                    },
                )

            return request_id

        except Exception as e:
            logger.error(f"Fast-reply execution failed: {e}")
            return self._handle_complex_workflow(request)

    def _handle_complex_workflow(self, request: RequestMetadata) -> str:
        """Existing complex workflow handling with duration tracking"""
        try:
            request_id = "wf_" + str(uuid.uuid4()).split("-")[-1]
            time.time()

            logger.info(f"Handling workflow '{request_id}' with Universal Agent")

            # Start duration tracking at the beginning of execution
            duration_logger = get_duration_logger()
            duration_logger.start_workflow_tracking(
                workflow_id=request_id,
                source=WorkflowSource.CLI,  # Will be updated by caller if needed
                workflow_type=WorkflowType.COMPLEX_WORKFLOW,
                instruction=request.prompt,
            )

            # Create task plan using Universal Agent with planning role
            task_context = self._create_task_plan(request.prompt, request_id)

            # Store the workflow context
            self.active_workflows[request_id] = task_context

            # Start execution and workflow state
            task_context.start_execution()
            if self.state == WorkflowState.IDLE:
                self.state = WorkflowState.RUNNING
                self.start_time = time.time()
                self.last_checkpoint_time = self.start_time

            # Execute DAG with parallel task processing
            self._execute_dag_parallel(task_context)

            logger.info(f"Workflow '{request_id}' created and started")
            return request_id

        except Exception as e:
            logger.error(f"Error handling workflow request: {e}")

            # Create a failed task context for error tracking
            try:
                # Create minimal task context for error tracking
                task_graph = TaskGraph(tasks=[])  # Empty task list for error tracking
                task_context = TaskContext(task_graph, context_id=request_id)
                task_context.fail_execution(f"Workflow failed: {str(e)}")

                # Store the failed workflow context
                self.active_workflows[request_id] = task_context

                return request_id  # Return ID even for failed workflows for tracking

            except Exception as context_error:
                logger.error(f"Error creating failed workflow context: {context_error}")
                return None

    def pause_workflow(self, workflow_id: Optional[str] = None) -> dict:
        """Pause workflow execution and create comprehensive checkpoint.

        Args:
            workflow_id: Optional specific workflow to pause, or all workflows

        Returns:
            Dict: Checkpoint data for resuming
        """
        if workflow_id:
            return self.pause_request(workflow_id)

        # Pause entire workflow engine
        self.state = WorkflowState.PAUSED

        checkpoint = {
            "workflow_engine_state": {
                "state": self.state.value,
                "start_time": self.start_time,
                "last_checkpoint_time": self.last_checkpoint_time,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "checkpoint_interval": self.checkpoint_interval,
            },
            "task_queue": [
                {
                    "priority": task.priority.value,
                    "scheduled_time": task.scheduled_time,
                    "task_id": task.task_id,
                    "context_id": task.context.context_id,
                }
                for task in self.task_queue
            ],
            "running_tasks": {
                task_id: {
                    "start_time": info["start_time"],
                    "context_id": info["context"].context_id,
                }
                for task_id, info in self.running_tasks.items()
            },
            "active_workflows": len(self.active_workflows),
            "timestamp": time.time(),
        }

        logger.info("WorkflowEngine paused and checkpoint created")
        return checkpoint

    def resume_workflow(
        self, workflow_id: Optional[str] = None, checkpoint: Optional[dict] = None
    ) -> bool:
        """Resume workflow execution from checkpoint.

        Args:
            workflow_id: Optional specific workflow to resume
            checkpoint: Optional checkpoint to resume from

        Returns:
            bool: True if resumed successfully
        """
        if workflow_id:
            return self.resume_request(workflow_id, checkpoint)

        # Resume entire workflow engine
        try:
            if checkpoint:
                # Restore workflow engine state
                engine_state = checkpoint.get("workflow_engine_state", {})
                self.start_time = engine_state.get("start_time", time.time())
                self.last_checkpoint_time = engine_state.get(
                    "last_checkpoint_time", time.time()
                )

                logger.info("WorkflowEngine resumed from checkpoint")
            else:
                logger.info("WorkflowEngine resumed without checkpoint")

            self.state = WorkflowState.RUNNING

            # Resume task queue processing
            self._process_task_queue()

            return True

        except Exception as e:
            logger.error(f"Error resuming WorkflowEngine: {e}")
            return False

    # ==================== DAG EXECUTION WITH PARALLEL PROCESSING ====================

    def _execute_dag_parallel(self, task_context: TaskContext):
        """Execute DAG with parallel task execution and concurrency control.

        Combines DAG traversal with priority-based task execution from TaskScheduler.

        Args:
            task_context: The task context containing the DAG to execute
        """
        ready_tasks = task_context.get_ready_tasks()

        for task in ready_tasks:
            if len(self.running_tasks) < self.max_concurrent_tasks:
                self._execute_task_async(task_context, task)
            else:
                # Queue task with priority
                self.schedule_task(task_context, task, TaskPriority.NORMAL)

    def _execute_task_async(self, task_context: TaskContext, task: TaskNode):
        r"""\1

        Args:
            task_context: The task context
            task: The task to execute
        """
        # Add to running tasks
        self.running_tasks[task.task_id] = {
            "task": task,
            "context": task_context,
            "start_time": time.time(),
            "priority": TaskPriority.NORMAL,
        }

        # Delegate task execution
        self.delegate_task(task_context, task)

    def schedule_task(
        self,
        context: TaskContext,
        task: TaskNode,
        priority: TaskPriority = TaskPriority.NORMAL,
    ):
        r"""\1

        Args:
            context: Task context
            task: Task node to schedule
            priority: Task priority level
        """
        queued_task = QueuedTask(
            priority=priority, scheduled_time=time.time(), task=task, context=context
        )

        # Add to priority queue (heapq maintains order)
        heapq.heappush(self.task_queue, queued_task)

        logger.info(f"Task '{task.task_id}' scheduled with priority {priority.name}")

        # Process queue if workflow is running
        if self.state == WorkflowState.RUNNING:
            self._process_task_queue()

    def _process_task_queue(self):
        """Process the task queue respecting concurrency limits."""
        while (
            len(self.running_tasks) < self.max_concurrent_tasks
            and self.task_queue
            and self.state == WorkflowState.RUNNING
        ):
            # Get highest priority task
            queued_task = heapq.heappop(self.task_queue)

            # Start task execution
            self._start_task_execution(queued_task)

    def _start_task_execution(self, queued_task: QueuedTask):
        r"""\1

        Args:
            queued_task: The task to start executing
        """
        task = queued_task.task
        context = queued_task.context

        # Add to running tasks
        self.running_tasks[task.task_id] = {
            "task": task,
            "context": context,
            "start_time": time.time(),
            "priority": queued_task.priority,
        }

        # Delegate task execution
        try:
            self.delegate_task(context, task)
            logger.info(
                f"Started execution of task '{task.task_id}' (priority: {queued_task.priority.name})"
            )
        except Exception as e:
            logger.error(f"Error starting task '{task.task_id}': {e}")
            # Remove from running tasks on error
            del self.running_tasks[task.task_id]

    # ==================== TASK DELEGATION AND EXECUTION ====================

    def _create_task_plan(
        self, instruction: str, request_id: Optional[str] = None
    ) -> TaskContext:
        r"""\1

        Args:
            instruction: User instruction to plan for
            request_id: Optional request ID for tracking

        Returns:
            TaskContext: Task context with planned tasks

        Raises:
            Exception: If task planning fails - no fallbacks, return error to user
        """
        # Use existing planning tools via Universal Agent
        # This follows the architecture: WorkflowEngine → Universal Agent → @tool functions
        from llm_provider.planning_tools import create_task_plan

        # Use the dynamic planning tool to create TaskGraph
        planning_result = create_task_plan(
            instruction=instruction, llm_factory=self.llm_factory, request_id=request_id
        )

        # Extract TaskGraph from planning result
        task_graph = planning_result.get("task_graph")
        tasks = planning_result.get("tasks", [])
        dependencies = planning_result.get("dependencies", [])

        if not task_graph or not tasks:
            raise ValueError("Planning tool did not return valid tasks and task_graph")

        # Convert TaskGraph to TaskContext using from_tasks method
        task_context = TaskContext.from_tasks(
            tasks=tasks, dependencies=dependencies, request_id=request_id
        )

        logger.info(f"Successfully created task plan with {len(tasks)} tasks")
        return task_context

    def delegate_task(self, task_context: TaskContext, task: TaskNode):
        r"""\1

        Args:
            task_context: The task context containing state
            task: The task node to execute
        """
        if task.status != TaskStatus.PENDING:
            return

        try:
            # Mark task as running
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()

            # Use agent_id as role name directly (selected by planning LLM)
            role_name = task.agent_id

            # Get LLM type from task node (set by planning LLM)
            llm_type_str = getattr(task, "llm_type", "DEFAULT")
            try:
                from llm_provider.factory import LLMType

                llm_type = LLMType(llm_type_str)
            except (ValueError, AttributeError):
                logger.warning(
                    f"Invalid LLM type '{llm_type_str}' for task '{task.task_id}', using DEFAULT"
                )
                llm_type = LLMType.DEFAULT

            logger.info(
                f"Delegating task '{task.task_id}' to role '{role_name}' with model type '{llm_type.value}'"
            )

            # Prepare task execution context with predecessor results
            execution_config = task_context.prepare_task_execution(task.task_id)

            # Get predecessor results from TaskGraph history
            predecessor_results = task_context.task_graph.get_task_history(task.task_id)

            # Enhanced prompt with predecessor results
            base_prompt = execution_config.get(
                "prompt", getattr(task, "prompt", "No prompt available")
            )

            # Only add predecessor context if there are meaningful results
            # Filter out empty results and "The beginning" placeholder
            meaningful_results = [
                result
                for result in predecessor_results
                if result and result.strip() and result != "The beginning"
            ]

            if meaningful_results:
                enhanced_prompt = f"""Previous task results available for context:
{chr(10).join(f"- {result}" for result in meaningful_results)}

Current task: {base_prompt}"""
            else:
                enhanced_prompt = base_prompt

            # Execute task using Universal Agent with enhanced context
            # If role is "None", Universal Agent will handle dynamic role generation
            result = self.universal_agent.execute_task(
                instruction=enhanced_prompt,
                role=role_name,
                llm_type=llm_type,
                context=task_context,
            )

            # Complete the task and get next ready tasks
            next_tasks = task_context.complete_task(task.task_id, result)

            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]

            # Delegate next ready tasks
            for _next_task in next_tasks:
                self._execute_dag_parallel(task_context)  # Re-evaluate DAG

            # Publish task completion message
            self.message_bus.publish(
                self,
                MessageType.TASK_RESPONSE,
                {
                    "request_id": task_context.task_graph.request_id,
                    "task_id": task.task_id,
                    "result": result,
                    "status": "completed",
                },
            )

        except Exception as e:
            logger.error(f"Error delegating task '{task.task_id}': {e}")
            self.handle_task_error(task_context, task, str(e))

    def _get_llm_type_for_role(self, role_name: str) -> LLMType:
        r"""\1

        Args:
            role_name: The role name

        Returns:
            LLMType: Appropriate semantic model type
        """
        # Try to get from role definition first
        role_def = self.role_registry.get_role(role_name)
        if role_def and "model_config" in role_def.config:
            model_type = role_def.config["model_config"].get("model_type")
            if model_type:
                try:
                    return LLMType(model_type.upper())
                except ValueError:
                    logger.warning(
                        f"Invalid model type '{model_type}' in role '{role_name}', using DEFAULT"
                    )

        # Default to DEFAULT model type - let planning agent decide complexity
        logger.info(f"No model type specified for role '{role_name}', using DEFAULT")
        return LLMType.DEFAULT

    def _is_simple_request(self, instruction: str) -> bool:
        r"""\1

        Args:
            instruction: The user instruction

        Returns:
            bool: True if request is simple
        """
        simple_patterns = [
            "what is",
            "calculate",
            "compute",
            "add",
            "subtract",
            "multiply",
            "divide",
            "weather in",
            "search for",
            "summarize",
            "send message",
        ]

        instruction_lower = instruction.lower()
        return any(pattern in instruction_lower for pattern in simple_patterns)

    def _parse_planning_result(
        self, plan_result: str, instruction: str, request_id: str
    ) -> TaskContext:
        r"""\1

        Args:
            plan_result: JSON string from planning agent
            instruction: Original user instruction
            request_id: Request ID for tracking

        Returns:
            TaskContext: Parsed task context with TaskGraph
        """
        try:
            import json

            # Try to parse the JSON from the planning result
            # The LLM might include extra text, so extract JSON
            plan_data = None

            # Look for JSON in the response
            if "{" in plan_result and "}" in plan_result:
                start_idx = plan_result.find("{")
                end_idx = plan_result.rfind("}") + 1
                json_str = plan_result[start_idx:end_idx]
                plan_data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in planning result")

            # Convert to TaskDescription objects
            tasks = []
            for task_data in plan_data.get("tasks", []):
                task = TaskDescription(
                    task_id=task_data.get("task_id", f"task_{len(tasks)+1}"),
                    task_name=task_data.get("task_name", "Unnamed Task"),
                    agent_id=task_data.get("agent_id", "planning_agent"),
                    task_type=task_data.get("task_type", "Execution"),
                    prompt=task_data.get("prompt", instruction),
                    request_id=request_id,
                )
                tasks.append(task)

            # Convert to TaskDependency objects
            dependencies = []
            for dep_data in plan_data.get("dependencies", []):
                dependency = TaskDependency(
                    source=dep_data.get("source"), target=dep_data.get("target")
                )
                dependencies.append(dependency)

            # Create TaskContext from parsed data
            task_context = TaskContext.from_tasks(
                tasks=tasks, dependencies=dependencies, request_id=request_id
            )

            logger.info(
                f"Successfully parsed planning result into {len(tasks)} tasks with {len(dependencies)} dependencies"
            )
            return task_context

        except Exception as e:
            logger.error(f"Failed to parse planning result: {e}")
            logger.debug(f"Planning result was: {plan_result}")

            # Fallback: create a single task for direct execution
            tasks = [
                TaskDescription(
                    task_name="Execute Request (Fallback)",
                    agent_id="planning_agent",
                    task_type="DirectExecution",
                    prompt=instruction,
                    request_id=request_id,
                )
            ]

            task_context = TaskContext.from_tasks(
                tasks=tasks, dependencies=[], request_id=request_id
            )

            return task_context

    # ==================== ERROR HANDLING AND RECOVERY ====================

    def handle_task_error(
        self, task_context: TaskContext, task: TaskNode, error_message: str
    ):
        r"""\1

        Args:
            task_context: The task context
            task: The failed task
            error_message: Error description
        """
        try:
            task.status = TaskStatus.FAILED
            task.stop_reason = error_message

            # Add error to task context metadata
            task_context.set_metadata(
                f"error_{task.task_id}",
                {
                    "error": error_message,
                    "timestamp": time.time(),
                    "retry_count": getattr(task, "retry_count", 0),
                },
            )

            # Implement retry logic
            retry_count = getattr(task, "retry_count", 0)
            if retry_count < self.max_retries:
                task.retry_count = retry_count + 1
                task.status = TaskStatus.PENDING

                logger.info(
                    f"Retrying task '{task.task_id}' (attempt {retry_count + 1}/{self.max_retries})"
                )
                time.sleep(self.retry_delay)

                # Retry the task
                self.delegate_task(task_context, task)
            else:
                logger.error(
                    f"Task '{task.task_id}' failed after {self.max_retries} retries: {error_message}"
                )
                task.status = TaskStatus.FAILED

                # Remove from running tasks
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]

                # Check if this causes request failure
                if self._should_fail_request(task_context):
                    task_context.fail_execution(
                        f"Critical task '{task.task_id}' failed: {error_message}"
                    )

        except Exception as e:
            logger.error(f"Error handling task error for '{task.task_id}': {e}")

    def handle_task_completion(self, completion_data: dict):
        r"""\1

        Args:
            completion_data: Task completion information
        """
        task_id = completion_data.get("task_id")
        if task_id in self.running_tasks:
            task_info = self.running_tasks.pop(task_id)
            execution_time = time.time() - task_info["start_time"]

            logger.info(f"Task '{task_id}' completed in {execution_time:.2f}s")

            # Process more tasks from queue
            self._process_task_queue()

            # Create checkpoint if interval elapsed
            self._maybe_create_checkpoint()

    def handle_task_error_event(self, error_data: dict):
        r"""\1

        Args:
            error_data: Task error information
        """
        task_id = error_data.get("task_id")
        if task_id in self.running_tasks:
            self.running_tasks.pop(task_id)
            error_message = error_data.get("error_message", "Unknown error")

            logger.error(f"Task '{task_id}' failed: {error_message}")

            # Process more tasks from queue
            self._process_task_queue()

    def _should_fail_request(self, task_context: TaskContext) -> bool:
        r"""\1

        Args:
            task_context: The task context to check

        Returns:
            bool: True if request should fail
        """
        try:
            failed_tasks = task_context.task_graph.get_failed_tasks()
            return len(failed_tasks) > 0
        except AttributeError:
            # Handle case where task_graph might not have get_failed_tasks method
            return False

    def _maybe_create_checkpoint(self):
        """Create checkpoint if interval has elapsed."""
        current_time = time.time()
        if (
            self.last_checkpoint_time
            and current_time - self.last_checkpoint_time >= self.checkpoint_interval
        ):
            self.pause_workflow()
            # In a full implementation, this would be persisted
            logger.info("Automatic checkpoint created")
            self.state = WorkflowState.RUNNING  # Resume after checkpoint
            self.last_checkpoint_time = current_time

    # ==================== PAUSE/RESUME FUNCTIONALITY ====================

    def pause_request(self, request_id: str) -> Optional[dict]:
        r"""\1

        Args:
            request_id: Request ID to pause

        Returns:
            Dict: Checkpoint for resuming execution, or None if not available
        """
        task_context = self.active_workflows.get(request_id)
        if not task_context:
            logger.error(f"Request '{request_id}' not found in contexts")
            return None

        try:
            checkpoint = task_context.pause_execution()
            logger.info(f"Request '{request_id}' paused")
            return checkpoint
        except Exception as e:
            logger.error(f"Error pausing request '{request_id}': {e}")
            return None

    def resume_request(
        self, request_id: str, checkpoint: Optional[dict] = None
    ) -> bool:
        r"""\1

        Args:
            request_id: Request ID to resume
            checkpoint: Optional checkpoint to resume from

        Returns:
            bool: True if resumed successfully
        """
        task_context = self.active_workflows.get(request_id)
        if not task_context:
            logger.error(f"Request '{request_id}' not found in contexts")
            return False

        try:
            task_context.resume_execution(checkpoint)

            # Resume ready tasks
            ready_tasks = task_context.get_ready_tasks()
            for task in ready_tasks:
                if task.status == TaskStatus.PENDING:
                    self.delegate_task(task_context, task)

            logger.info(
                f"Request '{request_id}' resumed with {len(ready_tasks)} ready tasks"
            )
            return True
        except Exception as e:
            logger.error(f"Error resuming request '{request_id}': {e}")
            return False

    # ==================== STATUS AND MONITORING ====================

    def get_workflow_metrics(self) -> dict[str, Any]:
        r"""\1

        Returns:
            Dict: Comprehensive workflow metrics
        """
        uptime = None
        if self.start_time:
            end_time = self.end_time or time.time()
            uptime = end_time - self.start_time

        return {
            # Workflow engine state
            "state": self.state,
            "uptime": uptime,
            # Task queue metrics (from TaskScheduler)
            "queued_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "checkpoint_interval": self.checkpoint_interval,
            "last_checkpoint_time": self.last_checkpoint_time,
            # Workflow tracking metrics (from RequestManager)
            "active_workflows": len(self.active_workflows),
            # Priority distribution
            "queue_priorities": [task.priority.name for task in self.task_queue],
            "running_task_priorities": [
                info["priority"].name for info in self.running_tasks.values()
            ],
            # Universal Agent metrics
            "universal_agent_status": self.get_universal_agent_status(),
        }

    def get_request_status(self, request_id: str) -> dict:
        r"""\1

        Args:
            request_id: Request ID to check

        Returns:
            Dict: Request status information
        """
        try:
            # Check unified active_workflows storage
            task_context = self.active_workflows.get(request_id)
            if not task_context:
                return {"error": f"Request '{request_id}' not found"}

            # Determine if it's a fast reply or complex workflow
            if request_id.startswith("fr_"):
                return self._get_fast_reply_status(request_id, task_context)
            else:
                return self._get_regular_workflow_status(request_id, task_context)

        except Exception as e:
            logger.error(f"Error getting request status for '{request_id}': {e}")
            return {"error": str(e)}

    def _get_fast_reply_status(self, request_id: str, task_context) -> dict:
        """Get status for fast reply requests from unified storage."""
        task_node = task_context.task_graph.nodes.get(request_id)
        if not task_node:
            return {"error": f"Fast reply task node not found for '{request_id}'"}

        # Extract metadata from task_context
        confidence = task_node.task_context.get("confidence", 0.0)
        role = task_node.role or task_node.agent_id

        return {
            "request_id": request_id,
            "execution_state": task_context.execution_state.value,
            "is_completed": task_context.execution_state.value == "COMPLETED",
            "result": task_node.result,
            "role": role,
            "confidence": confidence,
            "execution_time_ms": (
                (task_node.duration * 1000) if task_node.duration else 0
            ),
        }

    def _get_regular_workflow_status(self, request_id: str, task_context) -> dict:
        """Get status for regular workflow requests."""
        is_completed = task_context.is_completed()

        # Complete duration tracking for complex workflows when they finish
        if is_completed:
            self._complete_workflow_duration_tracking(request_id, task_context)

        return {
            "request_id": request_id,
            "execution_state": task_context.execution_state.value,
            "is_completed": is_completed,
            "performance_metrics": task_context.get_performance_metrics(),
            "task_statuses": {
                node_id: node.status.value
                for node_id, node in task_context.task_graph.nodes.items()
            },
        }

    def _complete_workflow_duration_tracking(self, request_id: str, task_context):
        """Complete duration tracking for regular workflows."""
        if not (
            hasattr(self, "duration_logger")
            and request_id in self.duration_logger.active_workflows
        ):
            return

        try:
            task_statuses = {
                node_id: node.status.value
                for node_id, node in task_context.task_graph.nodes.items()
            }
            task_count = len(task_statuses)
            failed_tasks = [
                status for status in task_statuses.values() if status == "FAILED"
            ]
            success = len(failed_tasks) == 0
            error_message = (
                f"{len(failed_tasks)} tasks failed" if failed_tasks else None
            )

            self.duration_logger.complete_workflow_tracking(
                workflow_id=request_id,
                success=success,
                error_message=error_message,
                task_count=task_count,
            )
        except Exception as e:
            logger.debug(f"Duration tracking already completed for {request_id}: {e}")

    def get_request_context(self, request_id: str) -> Optional[TaskContext]:
        r"""\1

        Args:
            request_id: Request ID

        Returns:
            TaskContext: Task context if available
        """
        return self.active_workflows.get(request_id)

    def get_universal_agent_status(self) -> dict:
        r"""\1

        Returns:
            Dict: Status information
        """
        return {
            "universal_agent_enabled": True,
            "has_llm_factory": self.llm_factory is not None,
            "has_universal_agent": self.universal_agent is not None,
            "mcp_integration": (
                self.universal_agent.get_mcp_status()
                if self.universal_agent
                else {"mcp_available": False}
            ),
            "active_contexts": len(self.active_workflows),
            "framework": self.llm_factory.get_framework() if self.llm_factory else None,
        }

    def get_queue_status(self) -> dict[str, Any]:
        r"""\1

        Returns:
            Dict: Queue status information
        """
        return {
            "total_queued": len(self.task_queue),
            "total_running": len(self.running_tasks),
            "available_slots": self.max_concurrent_tasks - len(self.running_tasks),
            "queue_by_priority": {
                priority.name: len(
                    [t for t in self.task_queue if t.priority == priority]
                )
                for priority in TaskPriority
            },
            "running_by_priority": {
                priority.name: len(
                    [
                        info
                        for info in self.running_tasks.values()
                        if info["priority"] == priority
                    ]
                )
                for priority in TaskPriority
            },
        }

    def list_active_requests(self) -> list[str]:
        r"""\1

        Returns:
            List[str]: List of active request IDs
        """
        return list(self.active_workflows.keys())

    def cleanup_completed_requests(self, max_age_seconds: int = 3600):
        r"""\1

        Args:
            max_age_seconds: Maximum age in seconds for completed requests
        """
        current_time = time.time()
        to_remove = []

        for request_id, context in self.active_workflows.items():
            if context.is_completed():
                # Check if context is old enough to clean up
                context_age = current_time - context.get_performance_metrics().get(
                    "start_time", current_time
                )
                if context_age > max_age_seconds:
                    to_remove.append(request_id)

        for request_id in to_remove:
            del self.active_workflows[request_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} completed workflows")

    def clear_queue(self):
        """Clear all queued tasks (running tasks continue)."""
        cleared_count = len(self.task_queue)
        self.task_queue.clear()
        logger.info(f"Cleared {cleared_count} queued tasks")

    def get_running_task_ids(self) -> list[str]:
        """Get list of currently running task IDs."""
        return list(self.running_tasks.keys())

    def get_queued_task_ids(self) -> list[str]:
        """Get list of queued task IDs."""
        return [task.task_id for task in self.task_queue]

    def start_workflow_engine(self):
        """Start the workflow engine."""
        self.state = WorkflowState.RUNNING
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        logger.info("WorkflowEngine started")

    def stop_workflow_engine(self):
        """Stop the workflow engine."""
        self.state = WorkflowState.STOPPED
        self.end_time = time.time()
        logger.info("WorkflowEngine stopped")

    # ==================== MCP INTEGRATION ====================

    def _initialize_mcp_manager(
        self, config_path: Optional[str] = None
    ) -> Optional[MCPClientManager]:
        r"""\1

        Args:
            config_path: Optional path to MCP configuration file.
                        If None, MCP integration is explicitly disabled.

        Returns:
            MCPClientManager instance or None if MCP not configured
        """
        # If config_path is explicitly None, MCP integration is disabled
        if config_path is None:
            logger.info("MCP integration explicitly disabled")
            return None

        # If config_path is empty string, try default paths
        if not config_path:
            default_paths = [
                "config/mcp_config.yaml",
                "mcp_config.yaml",
                os.path.expanduser("~/.config/generative-agent/mcp_config.yaml"),
            ]

            for path in default_paths:
                if os.path.exists(path):
                    config_path = path
                    break

        if not config_path or not os.path.exists(config_path):
            logger.info("No MCP configuration found, MCP integration disabled")
            return None

        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)

            manager = MCPClientManager()
            manager.load_servers_from_config(config_data)

            logger.info(
                "config_path=<%s>, servers=<%d> | MCP manager initialized successfully",
                config_path,
                len(manager.clients),
            )

            return manager

        except Exception as e:
            logger.error(
                "config_path=<%s>, error=<%s> | Failed to initialize MCP manager",
                config_path,
                str(e),
            )
            return None

    def get_mcp_tools(self, role: Optional[str] = None) -> list[dict]:
        r"""\1

        Args:
            role: Optional role to filter tools for

        Returns:
            List of available MCP tools
        """
        if not self.mcp_manager:
            return []

        return self.mcp_manager.get_tools_for_role(role or "default")

    def execute_mcp_tool(self, tool_name: str, parameters: dict) -> dict:
        r"""\1

        Args:
            tool_name: Name of the MCP tool to execute
            parameters: Parameters to pass to the tool

        Returns:
            Tool execution result
        """
        if not self.mcp_manager:
            return {"error": "MCP manager not available"}

        try:
            return self.mcp_manager.execute_tool(tool_name, parameters)
        except Exception as e:
            logger.error(f"Error executing MCP tool '{tool_name}': {e}")
            return {"error": str(e)}

    # ==================== ROLE-BASED TASK DELEGATION ====================

    def _determine_role_from_agent_id(self, agent_id: str) -> str:
        r"""\1

        Args:
            agent_id: Legacy agent ID (e.g., 'planning_agent', 'search_agent')

        Returns:
            str: Role name (e.g., 'planning', 'search')
        """
        # Dynamic role determination - remove '_agent' suffix if present
        if agent_id.endswith("_agent"):
            role = agent_id[:-6]  # Remove '_agent'
        else:
            role = agent_id

        return role

    def update_workflow_source(
        self,
        workflow_id: str,
        source: WorkflowSource,
        user_id: str = None,
        channel_id: str = None,
    ):
        r"""\1

        Args:
            workflow_id: The workflow ID to update
            source: The workflow source (CLI, SLACK, API)
            user_id: Optional user ID for Slack workflows
            channel_id: Optional channel ID for Slack workflows
        """
        if (
            hasattr(self, "duration_logger")
            and workflow_id in self.duration_logger.active_workflows
        ):
            try:
                workflow_metrics = self.duration_logger.active_workflows[workflow_id]
                workflow_metrics.source = source
                if user_id:
                    workflow_metrics.user_id = user_id
                if channel_id:
                    workflow_metrics.channel_id = channel_id
                logger.debug(f"Updated workflow {workflow_id} source to {source.value}")
            except Exception as e:
                logger.debug(f"Could not update workflow source for {workflow_id}: {e}")

    def _determine_llm_type_for_role(self, role: str) -> LLMType:
        r"""\1

        Args:
            role: Role name

        Returns:
            LLMType: Optimal LLM type for the role
        """
        # Get LLM type from role registry (dynamic from YAML definitions)
        if (
            hasattr(self, "universal_agent")
            and self.universal_agent
            and hasattr(self.universal_agent, "role_registry")
        ):
            llm_type_str = self.universal_agent.role_registry.get_role_llm_type(role)
            try:
                return LLMType[llm_type_str.upper()]
            except (KeyError, AttributeError):
                logger.warning(
                    f"Invalid LLM type '{llm_type_str}' for role '{role}', using DEFAULT"
                )
                return LLMType.DEFAULT
        else:
            # Fallback if role registry not available
            logger.warning(
                f"Role registry not available, using DEFAULT LLM type for role '{role}'"
            )
            return LLMType.DEFAULT
