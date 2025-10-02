import logging
import time
import uuid
import heapq
from typing import Dict, Optional, List, Any
from enum import Enum
from dataclasses import dataclass, field
import yaml
import os

from common.task_graph import TaskGraph, TaskNode, TaskStatus, TaskDescription, TaskDependency
from common.message_bus import MessageBus, MessageType
from common.request_model import Request, RequestMetadata
from common.task_context import TaskContext, ExecutionState
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.universal_agent import UniversalAgent
from llm_provider.mcp_client import MCPClientManager

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
        self.task_id = self.task.task_id
    
    def __lt__(self, other):
        """Priority queue ordering: higher priority first, then by scheduled time."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value  # Higher priority first
        return self.scheduled_time < other.scheduled_time  # Earlier scheduled time first


class WorkflowEngine:
    """
    Unified workflow management with DAG execution and state persistence.
    
    This consolidates RequestManager + TaskScheduler functionality into a single
    WorkflowEngine for simplified architecture and better performance.
    
    Combines:
    - Request lifecycle management (from RequestManager)
    - Task scheduling and priority queuing (from TaskScheduler)  
    - DAG execution with concurrency control
    - Universal Agent integration with StrandsAgent framework
    """
    
    def __init__(self, llm_factory: LLMFactory, message_bus: MessageBus,
                 max_retries: int = 3, retry_delay: float = 1.0,
                 max_concurrent_tasks: int = 5, checkpoint_interval: int = 300,
                 mcp_config_path: Optional[str] = None):
        """
        Initialize WorkflowEngine with Universal Agent, task scheduling, and MCP integration.
        
        Args:
            llm_factory: LLMFactory for creating Universal Agent
            message_bus: Message bus for task distribution
            max_retries: Maximum retry attempts for failed tasks
            retry_delay: Delay between retry attempts
            max_concurrent_tasks: Maximum concurrent task execution
            checkpoint_interval: Checkpoint creation interval in seconds
            mcp_config_path: Optional path to MCP configuration file
        """
        self.llm_factory = llm_factory
        self.message_bus = message_bus
        
        # Initialize MCP manager
        self.mcp_manager = self._initialize_mcp_manager(mcp_config_path)
        
        # Create Universal Agent with MCP support
        self.universal_agent = UniversalAgent(llm_factory, mcp_manager=self.mcp_manager)
        
        # Workflow tracking (consolidated from RequestManager + TaskScheduler)
        self.active_workflows: Dict[str, TaskContext] = {}
        self.request_map: Dict[str, Request] = {}  # For backward compatibility
        
        # Task scheduling (integrated from TaskScheduler)
        self.task_queue: List[QueuedTask] = []
        self.running_tasks: Dict[str, Dict] = {}
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
        
        # Subscribe to workflow events (consolidated subscriptions)
        self.message_bus.subscribe(self, MessageType.INCOMING_REQUEST, self.handle_request)
        self.message_bus.subscribe(self, MessageType.TASK_RESPONSE, self.handle_task_completion)
        self.message_bus.subscribe(self, MessageType.AGENT_ERROR, self.handle_task_error_event)
        
        # Backward compatibility
        self.request_contexts = self.active_workflows
        
        logger.info("WorkflowEngine initialized with Universal Agent, task scheduling, and MCP integration")
    
    # ==================== UNIFIED WORKFLOW INTERFACE ====================
    
    def start_workflow(self, instruction: str, source_id: str = "client", target_id: str = "supervisor") -> str:
        """
        Create and start a new workflow from user instruction.
        
        Args:
            instruction: User instruction to create workflow for
            source_id: Source identifier for the workflow
            target_id: Target identifier for the workflow
            
        Returns:
            str: Workflow ID for tracking
        """
        request = RequestMetadata(prompt=instruction, source_id=source_id, target_id=target_id)
        return self.handle_request(request)
    
    def handle_request(self, request: RequestMetadata) -> str:
        """
        Handle incoming request using Universal Agent and TaskContext.
        
        Args:
            request: The incoming request metadata
            
        Returns:
            str: Request ID for tracking
        """
        try:
            request_id = 'req_' + str(uuid.uuid4()).split('-')[-1]
            request_time = time.time()
            
            logger.info(f"Handling request '{request_id}' with Universal Agent")
            
            # Create task plan using Universal Agent with planning role
            task_context = self._create_task_plan(request.prompt, request_id)
            
            # Store the task context and create compatibility request
            self.active_workflows[request_id] = task_context
            self.request_contexts = self.active_workflows  # Backward compatibility
            self.request_map[request_id] = Request(request, task_context.task_graph)
            
            # Start execution and workflow state
            task_context.start_execution()
            if self.state == WorkflowState.IDLE:
                self.state = WorkflowState.RUNNING
                self.start_time = time.time()
                self.last_checkpoint_time = self.start_time
            
            # Execute DAG with parallel task processing
            self._execute_dag_parallel(task_context)
            
            logger.info(f"Request '{request_id}' created and workflow started")
            return request_id
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            
            # Create a failed task context for error tracking
            try:
                # Create minimal task context for error tracking
                task_graph = TaskGraph()
                task_context = TaskContext(task_graph, request_id=request_id)
                task_context.fail_execution(f"Request failed: {str(e)}")
                
                # Store the failed context
                self.active_workflows[request_id] = task_context
                self.request_contexts = self.active_workflows  # Backward compatibility
                
                return request_id  # Return ID even for failed requests for tracking
                
            except Exception as context_error:
                logger.error(f"Error creating failed request context: {context_error}")
                return None
    
    def pause_workflow(self, workflow_id: Optional[str] = None) -> Dict:
        """
        Pause workflow execution and create comprehensive checkpoint.
        
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
                "checkpoint_interval": self.checkpoint_interval
            },
            "task_queue": [
                {
                    "priority": task.priority.value,
                    "scheduled_time": task.scheduled_time,
                    "task_id": task.task_id,
                    "context_id": task.context.context_id
                }
                for task in self.task_queue
            ],
            "running_tasks": {
                task_id: {
                    "start_time": info["start_time"],
                    "context_id": info["context"].context_id
                }
                for task_id, info in self.running_tasks.items()
            },
            "active_workflows": len(self.active_workflows),
            "timestamp": time.time()
        }
        
        logger.info("WorkflowEngine paused and checkpoint created")
        return checkpoint
    
    def resume_workflow(self, workflow_id: Optional[str] = None, checkpoint: Optional[Dict] = None) -> bool:
        """
        Resume workflow execution from checkpoint.
        
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
                self.last_checkpoint_time = engine_state.get("last_checkpoint_time", time.time())
                
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
        """
        Execute DAG with parallel task execution and concurrency control.
        
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
        """
        Execute task asynchronously with concurrency tracking.
        
        Args:
            task_context: The task context
            task: The task to execute
        """
        # Add to running tasks
        self.running_tasks[task.task_id] = {
            "task": task,
            "context": task_context,
            "start_time": time.time(),
            "priority": TaskPriority.NORMAL
        }
        
        # Delegate task execution
        self.delegate_task(task_context, task)
    
    def schedule_task(self, context: TaskContext, task: TaskNode, 
                     priority: TaskPriority = TaskPriority.NORMAL):
        """
        Schedule a task for execution with priority queuing.
        
        Args:
            context: Task context
            task: Task node to schedule
            priority: Task priority level
        """
        queued_task = QueuedTask(
            priority=priority,
            scheduled_time=time.time(),
            task=task,
            context=context
        )
        
        # Add to priority queue (heapq maintains order)
        heapq.heappush(self.task_queue, queued_task)
        
        logger.info(f"Task '{task.task_id}' scheduled with priority {priority.name}")
        
        # Process queue if workflow is running
        if self.state == WorkflowState.RUNNING:
            self._process_task_queue()
    
    def _process_task_queue(self):
        """Process the task queue respecting concurrency limits."""
        while (len(self.running_tasks) < self.max_concurrent_tasks and 
               self.task_queue and 
               self.state == WorkflowState.RUNNING):
            
            # Get highest priority task
            queued_task = heapq.heappop(self.task_queue)
            
            # Start task execution
            self._start_task_execution(queued_task)
    
    def _start_task_execution(self, queued_task: QueuedTask):
        """
        Start execution of a queued task.
        
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
            "priority": queued_task.priority
        }
        
        # Delegate task execution
        try:
            self.delegate_task(context, task)
            logger.info(f"Started execution of task '{task.task_id}' (priority: {queued_task.priority.name})")
        except Exception as e:
            logger.error(f"Error starting task '{task.task_id}': {e}")
            # Remove from running tasks on error
            del self.running_tasks[task.task_id]
    
    # ==================== TASK DELEGATION AND EXECUTION ====================
    
    def _create_task_plan(self, instruction: str, request_id: Optional[str] = None) -> TaskContext:
        """
        Create task plan using Universal Agent with planning role.
        
        Args:
            instruction: User instruction to plan for
            request_id: Optional request ID for tracking
            
        Returns:
            TaskContext: Task context with planned tasks
        """
        try:
            # Use Universal Agent with planning role and strong model
            planning_prompt = f"""
            Create a detailed task plan for the following instruction: {instruction}
            
            Return a structured plan with tasks, dependencies, and agent assignments.
            Each task should specify:
            - task_id: unique identifier
            - task_name: descriptive name
            - task_description: what needs to be done
            - agent_id: which agent should handle this (planning_agent, search_agent, weather_agent, summarizer_agent, slack_agent)
            - dependencies: list of task_ids this task depends on
            """
            
            # Execute planning with strong model for complex reasoning
            plan_result = self.universal_agent.execute_task(
                task_prompt=planning_prompt,
                role="planning",
                llm_type=LLMType.STRONG,
                context=None
            )
            
            # For now, create a simple task structure
            # In a full implementation, this would parse the plan_result
            tasks = [
                TaskDescription(
                    task_name="Execute Request",
                    agent_id="planning_agent",
                    task_type="RequestExecution",
                    prompt=instruction
                )
            ]
            
            # Create TaskContext from tasks
            task_context = TaskContext.from_tasks(
                tasks=tasks,
                dependencies=[],
                request_id=request_id
            )
            
            # Add planning result to conversation history
            task_context.add_system_message(f"Planning completed: {plan_result}")
            
            return task_context
            
        except Exception as e:
            logger.error(f"Error creating task plan: {e}")
            
            # Create a fallback task context for error cases
            try:
                tasks = [
                    TaskDescription(
                        task_name="Handle Error",
                        agent_id="planning_agent", 
                        task_type="ErrorHandling",
                        prompt=f"Handle error: {str(e)}"
                    )
                ]
                
                task_context = TaskContext.from_tasks(
                    tasks=tasks,
                    dependencies=[],
                    request_id=request_id
                )
                
                task_context.add_system_message(f"Planning failed, created error handling task: {str(e)}")
                return task_context
                
            except Exception as fallback_error:
                logger.error(f"Error creating fallback task context: {fallback_error}")
                raise e  # Re-raise original error if fallback fails
    
    def delegate_task(self, task_context: TaskContext, task: TaskNode):
        """
        Delegate task using Universal Agent with role-based execution.
        
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
            
            # Determine role and LLM type from agent_id
            role = self._determine_role_from_agent_id(task.agent_id)
            llm_type = self._determine_llm_type_for_role(role)
            
            logger.info(f"Delegating task '{task.task_id}' to role '{role}' with model type '{llm_type.value}'")
            
            # Prepare task execution context
            execution_config = task_context.prepare_task_execution(task.task_id)
            
            # Execute task using Universal Agent
            result = self.universal_agent.execute_task(
                task_prompt=execution_config.get("prompt", getattr(task, "prompt", "No prompt available")),
                role=role,
                llm_type=llm_type,
                context=task_context
            )
            
            # Complete the task and get next ready tasks
            next_tasks = task_context.complete_task(task.task_id, result)
            
            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            # Delegate next ready tasks
            for next_task in next_tasks:
                self._execute_dag_parallel(task_context)  # Re-evaluate DAG
            
            # Publish task completion message
            self.message_bus.publish(self, MessageType.TASK_RESPONSE, {
                "request_id": task_context.task_graph.request_id,
                "task_id": task.task_id,
                "result": result,
                "status": "completed"
            })
            
        except Exception as e:
            logger.error(f"Error delegating task '{task.task_id}': {e}")
            self.handle_task_error(task_context, task, str(e))
    
    def _determine_role_from_agent_id(self, agent_id: str) -> str:
        """
        Map existing agent_id values to roles for Universal Agent.
        
        Args:
            agent_id: The agent ID from the task
            
        Returns:
            str: The corresponding role name
        """
        agent_to_role_map = {
            "planning_agent": "planning",
            "search_agent": "search", 
            "weather_agent": "weather",
            "summarizer_agent": "summarizer",
            "slack_agent": "slack"
        }
        return agent_to_role_map.get(agent_id, "default")
    
    def _determine_llm_type_for_role(self, role: str) -> LLMType:
        """
        Map roles to appropriate LLM types for cost/performance optimization.
        
        Args:
            role: The agent role
            
        Returns:
            LLMType: Recommended semantic model type
        """
        role_to_llm_type = {
            "planning": LLMType.STRONG,    # Complex reasoning needs powerful model
            "analysis": LLMType.STRONG,    # Complex analysis needs powerful model
            "coding": LLMType.STRONG,      # Code generation needs powerful model
            "search": LLMType.WEAK,        # Simple search can use cheaper model
            "weather": LLMType.WEAK,       # Simple lookup
            "summarizer": LLMType.DEFAULT, # Balanced model for text processing
            "slack": LLMType.DEFAULT,      # Conversational tasks
            "default": LLMType.DEFAULT     # Default fallback
        }
        return role_to_llm_type.get(role, LLMType.DEFAULT)
    
    # ==================== ERROR HANDLING AND RECOVERY ====================
    
    def handle_task_error(self, task_context: TaskContext, task: TaskNode, error_message: str):
        """
        Handle task errors with retry logic.
        
        Args:
            task_context: The task context
            task: The failed task
            error_message: Error description
        """
        try:
            task.status = TaskStatus.FAILED
            task.stop_reason = error_message
            
            # Add error to task context metadata
            task_context.set_metadata(f"error_{task.task_id}", {
                "error": error_message,
                "timestamp": time.time(),
                "retry_count": getattr(task, "retry_count", 0)
            })
            
            # Implement retry logic
            retry_count = getattr(task, "retry_count", 0)
            if retry_count < self.max_retries:
                task.retry_count = retry_count + 1
                task.status = TaskStatus.PENDING
                
                logger.info(f"Retrying task '{task.task_id}' (attempt {retry_count + 1}/{self.max_retries})")
                time.sleep(self.retry_delay)
                
                # Retry the task
                self.delegate_task(task_context, task)
            else:
                logger.error(f"Task '{task.task_id}' failed after {self.max_retries} retries: {error_message}")
                task.status = TaskStatus.FAILED
                
                # Remove from running tasks
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                
                # Check if this causes request failure
                if self._should_fail_request(task_context):
                    task_context.fail_execution(f"Critical task '{task.task_id}' failed: {error_message}")
                    
        except Exception as e:
            logger.error(f"Error handling task error for '{task.task_id}': {e}")
    
    def handle_task_completion(self, completion_data: Dict):
        """
        Handle task completion events from message bus.
        
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
    
    def handle_task_error_event(self, error_data: Dict):
        """
        Handle task error events from message bus.
        
        Args:
            error_data: Task error information
        """
        task_id = error_data.get("task_id")
        if task_id in self.running_tasks:
            task_info = self.running_tasks.pop(task_id)
            error_message = error_data.get("error_message", "Unknown error")
            
            logger.error(f"Task '{task_id}' failed: {error_message}")
            
            # Process more tasks from queue
            self._process_task_queue()
    
    def _should_fail_request(self, task_context: TaskContext) -> bool:
        """
        Determine if a failed task should cause the entire request to fail.
        
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
        if (self.last_checkpoint_time and 
            current_time - self.last_checkpoint_time >= self.checkpoint_interval):
            
            checkpoint = self.pause_workflow()
            # In a full implementation, this would be persisted
            logger.info("Automatic checkpoint created")
            self.state = WorkflowState.RUNNING  # Resume after checkpoint
            self.last_checkpoint_time = current_time
    
    # ==================== PAUSE/RESUME FUNCTIONALITY ====================
    
    def pause_request(self, request_id: str) -> Optional[Dict]:
        """
        Pause request execution and return checkpoint.
        
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
    
    def resume_request(self, request_id: str, checkpoint: Optional[Dict] = None) -> bool:
        """
        Resume request execution from checkpoint or current state.
        
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
            
            logger.info(f"Request '{request_id}' resumed with {len(ready_tasks)} ready tasks")
            return True
        except Exception as e:
            logger.error(f"Error resuming request '{request_id}': {e}")
            return False
    
    # ==================== STATUS AND MONITORING ====================
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """
        Get consolidated workflow metrics combining request tracking and task queue statistics.
        
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
            "total_workflows_processed": len(self.request_map),
            
            # Priority distribution
            "queue_priorities": [task.priority.name for task in self.task_queue],
            "running_task_priorities": [
                info["priority"].name for info in self.running_tasks.values()
            ],
            
            # Universal Agent metrics
            "universal_agent_status": self.get_universal_agent_status()
        }
    
    def get_request_status(self, request_id: str) -> Dict:
        """
        Get current status of a request.
        
        Args:
            request_id: Request ID to check
            
        Returns:
            Dict: Request status information
        """
        try:
            task_context = self.active_workflows.get(request_id)
            if not task_context:
                return {"error": f"Request '{request_id}' not found"}
            
            return {
                "request_id": request_id,
                "execution_state": task_context.execution_state.value,
                "is_completed": task_context.is_completed(),
                "performance_metrics": task_context.get_performance_metrics(),
                "task_statuses": {
                    node_id: node.status.value 
                    for node_id, node in task_context.task_graph.nodes.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting request status for '{request_id}': {e}")
            return {"error": str(e)}
    
    def get_request_context(self, request_id: str) -> Optional[TaskContext]:
        """
        Get TaskContext for a request.
        
        Args:
            request_id: Request ID
            
        Returns:
            TaskContext: Task context if available
        """
        return self.active_workflows.get(request_id)
    
    def get_universal_agent_status(self) -> Dict:
        """
        Get status of Universal Agent integration.
        
        Returns:
            Dict: Status information
        """
        return {
            "universal_agent_enabled": True,
            "has_llm_factory": self.llm_factory is not None,
            "has_universal_agent": self.universal_agent is not None,
            "mcp_integration": self.universal_agent.get_mcp_status() if self.universal_agent else {"mcp_available": False},
            "active_contexts": len(self.active_workflows),
            "framework": self.llm_factory.get_framework() if self.llm_factory else None
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get detailed queue status.
        
        Returns:
            Dict: Queue status information
        """
        return {
            "total_queued": len(self.task_queue),
            "total_running": len(self.running_tasks),
            "available_slots": self.max_concurrent_tasks - len(self.running_tasks),
            "queue_by_priority": {
                priority.name: len([t for t in self.task_queue if t.priority == priority])
                for priority in TaskPriority
            },
            "running_by_priority": {
                priority.name: len([info for info in self.running_tasks.values()
                                  if info["priority"] == priority])
                for priority in TaskPriority
            }
        }
    
    def list_active_requests(self) -> List[str]:
        """
        List all active request IDs.
        
        Returns:
            List[str]: List of active request IDs
        """
        return list(self.active_workflows.keys())
    
    def cleanup_completed_requests(self, max_age_seconds: int = 3600):
        """
        Clean up completed requests older than specified age.
        
        Args:
            max_age_seconds: Maximum age in seconds for completed requests
        """
        current_time = time.time()
        to_remove = []
        
        for request_id, context in self.active_workflows.items():
            if context.is_completed():
                # Check if context is old enough to clean up
                context_age = current_time - context.get_performance_metrics().get('start_time', current_time)
                if context_age > max_age_seconds:
                    to_remove.append(request_id)
        
        for request_id in to_remove:
            del self.active_workflows[request_id]
            if request_id in self.request_map:
                del self.request_map[request_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} completed requests")
    
    def clear_queue(self):
        """Clear all queued tasks (running tasks continue)."""
        cleared_count = len(self.task_queue)
        self.task_queue.clear()
        logger.info(f"Cleared {cleared_count} queued tasks")
    
    def get_running_task_ids(self) -> List[str]:
        """Get list of currently running task IDs."""
        return list(self.running_tasks.keys())
    
    def get_queued_task_ids(self) -> List[str]:
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
    
    def _initialize_mcp_manager(self, config_path: Optional[str] = None) -> Optional[MCPClientManager]:
        """
        Initialize MCP client manager from configuration.
        
        Args:
            config_path: Optional path to MCP configuration file
            
        Returns:
            MCPClientManager instance or None if MCP not configured
        """
        if not config_path:
            # Try default config paths
            default_paths = [
                "config/mcp_config.yaml",
                "mcp_config.yaml",
                os.path.expanduser("~/.config/generative-agent/mcp_config.yaml")
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        if not config_path or not os.path.exists(config_path):
            logger.info("No MCP configuration found, MCP integration disabled")
            return None
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            manager = MCPClientManager()
            manager.load_servers_from_config(config_data)
            
            logger.info(
                "config_path=<%s>, servers=<%d> | MCP manager initialized successfully",
                config_path, len(manager.clients)
            )
            
            return manager
            
        except Exception as e:
            logger.error(
                "config_path=<%s>, error=<%s> | Failed to initialize MCP manager",
                config_path, str(e)
            )
            return None
    
    def get_mcp_tools(self, role: Optional[str] = None) -> List[Dict]:
        """
        Get available MCP tools for a role.
        
        Args:
            role: Optional role to filter tools for
            
        Returns:
            List of available MCP tools
        """
        if not self.mcp_manager:
            return []
        
        return self.mcp_manager.get_tools_for_role(role or "default")
    
    def execute_mcp_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """
        Execute an MCP tool with given parameters.
        
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


# Backward compatibility alias
RequestManager = WorkflowEngine