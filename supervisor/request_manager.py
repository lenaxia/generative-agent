import logging
import time
import uuid
from typing import Dict, Optional

from pydantic import Field
from common.task_graph import TaskGraph, TaskNode, TaskStatus
from supervisor.agent_manager import AgentManager
from supervisor.supervisor_config import SupervisorConfig
from common.message_bus import MessageBus, MessageType
from common.request_model import Request, RequestMetadata

logger = logging.getLogger(__name__)

class RequestManager:
    agent_manager: AgentManager = Field(..., description="Agent manager instance.")
    request_map: Dict[str, Request] = Field(default_factory=dict, description="Map of request IDs to task graphs.")
    message_bus: MessageBus = Field(..., description="Message Bus instance")
    config: SupervisorConfig = Field(..., description="Agent manager configuration")
    
    def __init__(self, agent_manager, message_bus):
        self.agent_manager = agent_manager
        self.message_bus = message_bus
        self.config = agent_manager.config
        self.request_map = {}
        
        self.message_bus.subscribe(self, MessageType.INCOMING_REQUEST, self.handle_request)

    def handle_request(self, request: RequestMetadata) -> str:
        try:
            request_id = 'req_' + str(uuid.uuid4()).split('-')[-1]
            request_time = time.time()

            # Create the task graph using the Planning Agent
            task_graph = self.create_task_graph(request.prompt, request_id)
            task_graph.request_id = request_id

            self.request_map[request_id] = Request(request, task_graph) 

            # Delegate the initial tasks
            for task in task_graph.get_entrypoint_nodes():
                self.delegate_task(task_graph, task)

            # TODO: This was causing some problems so commented it out for now.
            self.config.metrics_manager.update_metrics(request_id, {
                "start_time": time.time(),
                "tasks_completed": 0,
                "tasks_failed": 0,
                "retries": 0,
            })
            self.config.metrics_manager.update_metrics(request_id, {"start_time": request_time})
            logger.info(f"Received new request with ID '{request_id}'.")
            return request_id
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            raise e

    def create_task_graph(self, instruction: str, request_id: Optional[str] = None) -> TaskGraph:
        planning_start_time = time.time()
        planning_agent = self.agent_manager.get_agent("planning_agent")
        if planning_agent is None:
            logger.error("No planning agent found")
            raise Exception("No planning agent found")

        if planning_agent:
            planning_agent.set_agent_manager(self.agent_manager)
            
            task_data = {
                "task_id": "plan_" + str(uuid.uuid4()).split('-')[-1],
                "request_id": request_id,
                "agent_id": "planning_agent",
                "task_name": "Planning",
                "task_description": "Create a plan for the user request",
                "task_type": "RequestPlanning",
                "status": TaskStatus.PENDING,
                "prompt": instruction,
            }
            task_graph = planning_agent.run(task_data)
            self.config.metrics_manager.update_metrics(request_id, {"planning_duration": time.time() - planning_start_time})
            return task_graph
        else:
            logger.error("Planning Agent not found in the agent registry.")
            return None

    def delegate_task(self, task_graph: TaskGraph, task: TaskNode):
        if (task.status == TaskStatus.PENDING):
            try:
                task.status = TaskStatus.RUNNING
                task.start_time = time.time()
                task_data = task.model_dump()
                task_data["history"] = task_graph.get_task_history(task.task_id)
                self.message_bus.publish(self, MessageType.TASK_ASSIGNMENT, task_data)
            except Exception as e:
                logger.error(f"Error delegating task '{task.task_id}' for request '{task_graph.request_id}': {e}")
                self.handle_agent_error({"request_id": task_graph.request_id, "task_id": task.task_id, "error_message": str(e)})

    def monitor_progress(self, request_id: str, verbose: Optional[bool] = False):
        try:
            task_graph = self.request_map.get(request_id).task_graph
            if task_graph is None:
                logger.error(f"Request '{request_id}' not found.")
                return {"status": "error", "error": f"Request '{request_id}' not found."}

            start_time = task_graph.start_time  # Add a start_time attribute to the task_graph

            # Check task dependencies and conditional execution
            completed_nodes = []
            failed_nodes = []
            for node_id in task_graph.nodes:
                task = task_graph.nodes.get(node_id, None)
                if task.status == TaskStatus.COMPLETED:
                    completed_nodes.append(task)
                    self.config.metrics_manager.delta_metrics(request_id, {"tasks_completed": 1})
                    self.delegate_next_tasks(task_graph, task)
                elif task.status in (TaskStatus.FAILED, TaskStatus.RETRIESEXCEEDED):
                    failed_nodes.append(task)

            if task_graph.is_complete():
                self.handle_request_completion(request_id)
                
            if verbose:
                completed = task_graph.is_complete()
                num_nodes = len(task_graph.nodes)
                num_completed = len(completed_nodes)
                percent_completed = (num_completed / num_nodes) * 100 if num_nodes > 0 else 0
                runtime_duration = time.time() - start_time

                verbose_output = {
                    "status": completed,
                    "num_nodes": num_nodes,
                    "num_completed": num_completed,
                    "percent_completed": percent_completed,
                    "runtime_duration": runtime_duration,
                    "failed_nodes": [{"name": node.task_name, "prompt": node.prompt_template_formatted} for node in failed_nodes],
                    "node_info": task,
                }
                return verbose_output

        except Exception as e:
            logger.error(f"Error monitoring progress for request '{request_id}': {e}")
            self.handle_request_failure(request_id, str(e))

    def check_conditions(self, node: TaskNode, task_graph: TaskGraph) -> bool:
        # Check the conditions for executing the node
        # based on the task graph and previous task outputs
        # Return True if conditions are met, False otherwise
        return True  # Placeholder implementation

    def handle_task_response(self, response: Dict):
        """
        Handle a task response by updating the task graph and delegating the next tasks.

        Args:
            response (Dict): The task response containing the request ID, task ID, result and status.

        Raises:
            Exception: If an error occurs while handling the task response.
        """
        try:
            request_id = response.get("request_id")
            request = self.request_map[request_id]
            task_graph = request.task_graph
            
            if task_graph is None:
                logger.error(f"Request '{request_id}' not found.")
                return

            task_id = response.get("task_id")
            task = task_graph.get_node_by_task_id(task_id)
            if task is None:
                logger.error(f"Task '{task_id}' not found in the task graph.")
                return
            
            task.result = response.get("result", None)
            task.status = TaskStatus.COMPLETED
            task.duration = time.time() - task.start_time
            task_graph.history.append(response.get("result"))

            self.monitor_progress(request_id)
        except Exception as e:
            logger.error(f"Error handling task response for request '{request_id}', task '{task_id}': {e}")
            self.handle_agent_error({"request_id": request_id, "task_id": task_id, "error_message": str(e)})

    def delegate_next_tasks(self, task_graph: TaskGraph, node: TaskNode):
        """
        Delegates the next tasks in the task graph after the current task has been completed.

        :param node: The completed task node
        :return: None
        """
        for child_node in task_graph.get_child_nodes(task_graph.nodes, node):
            if self.all_parents_completed(task_graph.nodes, child_node.task_id):
                if self.check_conditions(child_node, task_graph):
                    self.delegate_task(task_graph, child_node)

    def all_parents_completed(self, nodes: Dict[str, TaskNode], node_id: str):
        """
        Checks if all parent nodes of the given node have been completed.

        :param node: The node to check
        :return: True if all parents have been completed, False otherwise
        """
        node = nodes[node_id]
        for edge in node.inbound_edges:
            parent_node = nodes[edge.source_id]
            if parent_node.status != TaskStatus.COMPLETED:
                return False
        return True

    def handle_agent_error(self, error: Dict):
        """
        Handles an agent error by updating the task graph and retrying the failed task.

        Args:
            error (Dict): The error data containing the request ID, task ID and error message.

        Raises:
            Exception: If an error occurs while handling the agent error.
        """
        try:
            request_id = error.get("request_id")
            task_graph = self.request_map.get(request_id).task_graph
            if task_graph is None:
                logger.error(f"Request '{request_id}' not found.")
                return

            task_id = error.get("task_id")
            node = task_graph.get_node_by_task_id(task_id)
            if node is None:
                logger.error(f"Task '{task_id}' not found in the task graph.")
                return

            node.status = TaskStatus.FAILED
            node.stop_reason = error.get("stop_reason")

            self.config.metrics_manager.update_metrics(request_id, {"tasks_failed": self.config.metrics_manager.get_metrics(request_id)["tasks_failed"] + 1})
            self.retry_failed_task(task_id, request_id)
        except Exception as e:
            logger.error(f"Error handling agent error for request '{request_id}', task '{task_id}': {e}")
            self.handle_request_failure(request_id, str(e))

    def retry_failed_task(self, task_id: str, request_id: str):
        try:
            task_graph = self.request_map.get(request_id).task_graph
            if task_graph is None:
                logger.error(f"Request '{request_id}' not found.")
                return

            node = task_graph.get_node_by_task_id(task_id)
            if node is None:
                logger.error(f"Task '{task_id}' not found in the task graph.")
                return

            request_metrics = self.config.metrics_manager.get_metrics(request_id)
            retries = request_metrics.get("retries", 0)
            if retries >= self.config.max_retries:
                node.status = TaskStatus.FAILED
                node.stop_reason = "Maximum retries reached"
                logger.error(f"Maximum retries ({self.config.max_retries}) reached for task '{task_id}'. Aborting request '{request_id}'.")
                self.handle_request_failure(request_id, f"Maximum retries reached for task '{task_id}'")
                return

            logger.info(f"Retrying task '{task_id}' (attempt {retries + 1}/{self.config.max_retries}).")
            self.config.metrics_manager.update_metrics(request_id, {"retries": retries + 1})
            time.sleep(self.config.retry_delay)
            self.delegate_task(node, request_id)
        except Exception as e:
            logger.error(f"Error retrying failed task '{task_id}' for request '{request_id}': {e}")
            self.handle_request_failure(request_id, str(e))

    def get_request_status(self, request_id: str) -> Dict[str, TaskStatus]:
        try:
            task_graph = self.request_map.get(request_id).task_graph
            if task_graph is None:
                logger.error(f"Request '{request_id}' not found.")
                return {}
    
            return {node: task_graph.nodes[node].status for node in task_graph.nodes}
        except Exception as e:
            logger.error(f"Error getting request status for '{request_id}': {e}")
            return {}
    
    def handle_request_completion(self, request_id: str):
        try:
            logger.info(f"Request '{request_id}' completed successfully.")
            end_time = time.time()
            duration = end_time - self.config.metrics_manager.get_metrics(request_id)["start_time"]
            self.config.metrics_manager.update_metrics(request_id, {"duration": duration})
            # Persist the request data and results
            self.persist_request(request_id)

            # Get the terminal nodes and print the results
            terminal_nodes = self.request_map[request_id].task_graph.get_terminal_nodes()
            results = {}
            for node in terminal_nodes:
                results[node.task_name] = node.result
            print(f"Results for request '{request_id}': {results}")

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
            task_graph = self.request_map[request_id].task_graph
            include_fields = {
                'task_name',
                'agent_id',
                'task_type',
                'prompt',
                'status',
                'inbound_edges',
                'outbound_edges',
                'result',
                'stop_reason',
                'start_time',
                'duration'
            }

            nodes = {node_id: node.model_dump(include=include_fields) for node_id, node in task_graph.nodes.items()}
            edges = {edge.source_id + '_' + edge.target_id: {"source": edge.source_id, "target": edge.target_id, "condition": edge.condition} for edge in task_graph.edges}
            
            request_data = {
                "request_id": request_id,
                "task_graph": {
                    "nodes": nodes,
                    "edges": edges,

                },
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
