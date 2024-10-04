import logging
import time
import uuid
from typing import Dict, Optional, Any

from pydantic import BaseModel, ValidationError, Field
from supervisor.task_graph import TaskGraph, TaskNode
from supervisor.task_models import Task, TaskStatus
from shared_tools.message_bus import MessageBus, MessageType

logger = logging.getLogger(__name__)

class RequestModel(BaseModel):
    instructions: str
    additional_data: Optional[Dict] = None

class RequestManager(BaseModel):
    config: Any = Field(..., description="Configuration object for the request manager.")
    request_map: Dict[str, TaskGraph] = Field(default_factory=dict, description="Map of request IDs to task graphs.")

    def __init__(self, config):
        super().__init__(config=config, request_map={})


    def handle_request(self, request: RequestModel) -> str:
        try:
            request_id = str(uuid.uuid4())
    
            # Get the list of available agents
            agents = self.config.agent_manager.get_agents()
    
            # Create the task graph using the Planning Agent
            task_graph = self.config.supervisor.create_task_graph(request.instructions, agents)
    
            # Delegate the initial tasks
            for task in task_graph.tasks:
                self.delegate_task(task, request_id)
    
            self.request_map[request_id] = task_graph
            self.config.metrics_manager.update_metrics(request_id, {
                "start_time": time.time(),
                "tasks_completed": 0,
                "tasks_failed": 0,
                "retries": 0,
            })
            logger.info(f"Received new request with ID '{request_id}'.")
            return request_id
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            raise e

    def monitor_progress(self, request_id: str):
        try:
            task_graph = self.request_map.get(request_id)
            if task_graph is None:
                logger.error(f"Request '{request_id}' not found.")
                return

            # Check task dependencies and conditional execution
            for node in task_graph.nodes:
                if node.status == TaskStatus.COMPLETED:
                    self.config.metrics_manager.update_metrics(request_id, {"tasks_completed": self.config.metrics_manager.get_metrics(request_id)["tasks_completed"] + 1})
                    for child_node in task_graph.get_child_nodes(node):
                        if self.check_conditions(child_node, task_graph):
                            self.delegate_task(child_node.task, request_id)

            if task_graph.is_complete():
                self.handle_request_completion(request_id)
        except Exception as e:
            logger.error(f"Error monitoring progress for request '{request_id}': {e}")
            self.handle_request_failure(request_id, str(e))

    def check_conditions(self, node: TaskNode, task_graph: TaskGraph) -> bool:
        # Check the conditions for executing the node
        # based on the task graph and previous task outputs
        # Return True if conditions are met, False otherwise
        return True  # Placeholder implementation

    def handle_task_response(self, response: Dict):
        try:
            request_id = response.get("request_id")
            task_graph = self.request_map.get(request_id)
            if task_graph is None:
                logger.error(f"Request '{request_id}' not found.")
                return

            task_id = response.get("task_id")
            node = task_graph.get_node_by_task_id(task_id)
            if node is None:
                logger.error(f"Task '{task_id}' not found in the task graph.")
                return

            node.task.status = TaskStatus.COMPLETED
            node.task.result = response.get("result")

            self.monitor_progress(request_id)
        except Exception as e:
            logger.error(f"Error handling task response for request '{request_id}', task '{task_id}': {e}")
            self.handle_agent_error({"request_id": request_id, "task_id": task_id, "error_message": str(e)})

    def handle_agent_error(self, error: Dict):
        try:
            request_id = error.get("request_id")
            task_graph = self.request_map.get(request_id)
            if task_graph is None:
                logger.error(f"Request '{request_id}' not found.")
                return

            task_id = error.get("task_id")
            node = task_graph.get_node_by_task_id(task_id)
            if node is None:
                logger.error(f"Task '{task_id}' not found in the task graph.")
                return

            node.task.status = TaskStatus.FAILED
            node.task.error = error.get("error_message")

            self.config.metrics_manager.update_metrics(request_id, {"tasks_failed": self.config.metrics_manager.get_metrics(request_id)["tasks_failed"] + 1})
            self.retry_failed_task(task_id, request_id)
        except Exception as e:
            logger.error(f"Error handling agent error for request '{request_id}', task '{task_id}': {e}")
            self.handle_request_failure(request_id, str(e))

    def retry_failed_task(self, task_id: str, request_id: str):
        try:
            task_graph = self.request_map.get(request_id)
            if task_graph is None:
                logger.error(f"Request '{request_id}' not found.")
                return

            node = task_graph.get_node_by_task_id(task_id)
            if node is None:
                logger.error(f"Task '{task_id}' not found in the task graph.")
                return

            retries = self.config.metrics_manager.get_metrics(request_id)["retries"]
            if retries >= self.config.max_retries:
                logger.error(f"Maximum retries ({self.config.max_retries}) reached for task '{task_id}'. Aborting request '{request_id}'.")
                self.handle_request_failure(request_id, f"Maximum retries reached for task '{task_id}'")
                return

            logger.info(f"Retrying task '{task_id}' (attempt {retries + 1}/{self.config.max_retries}).")
            self.config.metrics_manager.update_metrics(request_id, {"retries": retries + 1})
            time.sleep(self.config.retry_delay)
            self.delegate_task(node.task, request_id)
        except Exception as e:
            logger.error(f"Error retrying failed task '{task_id}' for request '{request_id}': {e}")
            self.handle_request_failure(request_id, str(e))

    def get_request_status(self, request_id: str) -> Dict[str, TaskStatus]:
        try:
            task_graph = self.request_map.get(request_id)
            if task_graph is None:
                logger.error(f"Request '{request_id}' not found.")
                return {}

            return {task.id: task.status for task in task_graph.tasks}
        except Exception as e:
            logger.error(f"Error getting request status for '{request_id}': {e}")
            return {}

    def delegate_task(self, task: Task, request_id: str):
        try:
            agent = self.config.agent_manager.get_agent(task.agent_id)
            if agent is None:
                logger.error(f"Agent '{task.agent_id}' not found in the agent registry.")
                return

            llm_client = self.config.llm_client_factory.get_client(task.agent_id)
            if llm_client is None:
                logger.error(f"No LLM client found for agent '{task.agent_id}'.")
                return

            tools = [tool.load() for tool in task.tools]
            agent.initialize_responder(task.prompt_template, tools)

            task_data = {
                "task_id": task.id,
                "agent_id": task.agent_id,
                "task_type": task.task_type,
                "prompt": task.prompt_template.format_prompt(**task.prompt_args),
                "request_id": request_id,
                "llm_client": llm_client,  # Pass the LLM client instance to the agent
            }
            self.config.supervisor.message_bus.publish(self, MessageType.TASK_ASSIGNMENT, task_data)
        except Exception as e:
            logger.error(f"Error delegating task '{task.id}' for request '{request_id}': {e}")
            self.handle_agent_error({"request_id": request_id, "task_id": task.id, "error_message": str(e)})

    def handle_request_completion(self, request_id: str):
        try:
            logger.info(f"Request '{request_id}' completed successfully.")
            end_time = time.time()
            duration = end_time - self.config.metrics_manager.get_metrics(request_id)["start_time"]
            self.config.metrics_manager.update_metrics(request_id, {"duration": duration})
            # Persist the request data and results
            self.persist_request(request_id)
        except Exception as e:
            logger.error(f"Error handling request completion for '{request_id}': {e}")

    def handle_request_failure(self, request_id: str, error_message: str):
        try:
            logger.error(f"Request '{request_id}' failed with error: {error_message}")
            end_time = time.time()
            duration = end_time - self.config.metrics_manager.get_metrics(request_id)["start_time"]
            self.config.metrics_manager.update_metrics(request_id, {"duration": duration, "error": error_message})
            # Persist the request data and error information
            self.persist_request(request_id)
        except Exception as e:
            logger.error(f"Error handling request failure for '{request_id}': {e}")

    def persist_request(self, request_id: str):
        try:
            # Implement logic to persist the request data, task graph, and results
            # to a persistent storage (e.g., database, file system)
            request_data = {
                "request_id": request_id,
                "task_graph": self.request_map[request_id].to_dict(),
                "metrics": self.config.metrics_manager.get_metrics(request_id),
            }
            self.config.metrics_manager.persist_metrics(request_id, request_data)
            logger.info(f"Persisted request '{request_id}'.")
        except Exception as e:
            logger.error(f"Error persisting request '{request_id}': {e}")

    def load_request(self, request_id: str) -> Dict:
        try:
            request_data = self.config.metrics_manager.load_metrics(request_id)
            if not request_data:
                logger.error(f"Request '{request_id}' not found in persistent storage.")
                return {}

            logger.info(f"Loaded request '{request_id}'.")
            return request_data
        except Exception as e:
            logger.error(f"Error loading request '{request_id}': {e}")
            return {}
