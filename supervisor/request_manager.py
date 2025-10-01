import logging
import time
import uuid
from typing import Dict, Optional, List
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


class RequestManager:
    """
    Modern RequestManager using Universal Agent with StrandsAgent framework.
    
    This replaces the legacy LangChain-based agent system with a unified
    Universal Agent approach for better performance and maintainability.
    """
    
    def __init__(self, llm_factory: LLMFactory, message_bus: MessageBus,
                 max_retries: int = 3, retry_delay: float = 1.0,
                 mcp_config_path: Optional[str] = None):
        """
        Initialize RequestManager with Universal Agent and MCP integration.
        
        Args:
            llm_factory: LLMFactory for creating Universal Agent
            message_bus: Message bus for task distribution
            max_retries: Maximum retry attempts for failed tasks
            retry_delay: Delay between retry attempts
            mcp_config_path: Optional path to MCP configuration file
        """
        self.llm_factory = llm_factory
        self.message_bus = message_bus
        
        # Initialize MCP manager
        self.mcp_manager = self._initialize_mcp_manager(mcp_config_path)
        
        # Create Universal Agent with MCP support
        self.universal_agent = UniversalAgent(llm_factory, mcp_manager=self.mcp_manager)
        
        # Request tracking
        self.request_contexts: Dict[str, TaskContext] = {}
        self.request_map: Dict[str, Request] = {}  # For compatibility
        
        # Configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Subscribe to incoming requests
        self.message_bus.subscribe(self, MessageType.INCOMING_REQUEST, self.handle_request)
        
        logger.info("RequestManager initialized with Universal Agent (StrandsAgent framework) and MCP integration")
    
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
            self.request_contexts[request_id] = task_context
            self.request_map[request_id] = Request(request, task_context.task_graph)
            
            # Start execution
            task_context.start_execution()
            
            # Delegate initial ready tasks
            ready_tasks = task_context.get_ready_tasks()
            for task in ready_tasks:
                self.delegate_task(task_context, task)
            
            logger.info(f"Request '{request_id}' created with {len(ready_tasks)} initial tasks")
            return request_id
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            raise e
    
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
            raise e
    
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
            
            # Delegate next ready tasks
            for next_task in next_tasks:
                self.delegate_task(task_context, next_task)
            
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
                
                # Check if this causes request failure
                if self._should_fail_request(task_context):
                    task_context.fail_execution(f"Critical task '{task.task_id}' failed: {error_message}")
                    
        except Exception as e:
            logger.error(f"Error handling task error for '{task.task_id}': {e}")
    
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
    
    # Pause/Resume Functionality
    
    def pause_request(self, request_id: str) -> Optional[Dict]:
        """
        Pause request execution and return checkpoint.
        
        Args:
            request_id: Request ID to pause
            
        Returns:
            Dict: Checkpoint for resuming execution, or None if not available
        """
        task_context = self.request_contexts.get(request_id)
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
        task_context = self.request_contexts.get(request_id)
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
    
    # Status and Monitoring
    
    def get_request_status(self, request_id: str) -> Dict:
        """
        Get current status of a request.
        
        Args:
            request_id: Request ID to check
            
        Returns:
            Dict: Request status information
        """
        try:
            task_context = self.request_contexts.get(request_id)
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
        return self.request_contexts.get(request_id)
    
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
            "active_contexts": len(self.request_contexts),
            "framework": self.llm_factory.get_framework() if self.llm_factory else None
        }
    
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
            
        Raises:
            ValueError: If MCP manager not available or tool not found
        """
        if not self.mcp_manager:
            raise ValueError("MCP integration not available")
        
        return self.mcp_manager.execute_tool(tool_name, parameters)
    
    def list_active_requests(self) -> List[str]:
        """
        Get list of active request IDs.
        
        Returns:
            List[str]: List of active request IDs
        """
        return [
            request_id for request_id, context in self.request_contexts.items()
            if context.execution_state in [ExecutionState.RUNNING, ExecutionState.PAUSED]
        ]
    
    def cleanup_completed_requests(self, max_age_seconds: int = 3600):
        """
        Clean up completed requests older than specified age.
        
        Args:
            max_age_seconds: Maximum age in seconds for completed requests
        """
        current_time = time.time()
        to_remove = []
        
        for request_id, context in self.request_contexts.items():
            if (context.execution_state == ExecutionState.COMPLETED and 
                context.end_time and 
                current_time - context.end_time > max_age_seconds):
                to_remove.append(request_id)
        
        for request_id in to_remove:
            del self.request_contexts[request_id]
            if request_id in self.request_map:
                del self.request_map[request_id]
            logger.info(f"Cleaned up completed request '{request_id}'")
