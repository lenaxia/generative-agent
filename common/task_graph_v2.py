from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from langchain_core.language_models import BaseChatModel
import uuid
import logging
import time
import yaml

from pydantic import BaseModel

logger = logging.getLogger("supervisor")

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRIESEXCEEDED = "RETRIESEXCEEDED"

    def __repr__(self):
        return f"TaskStatus.{self.name}"

yaml.SafeDumper.add_multi_representer(
    TaskStatus,
    yaml.representer.SafeRepresenter.represent_str,
)

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: Optional[dict] = None
    stop_reason: Optional[str] = None

class TaskDescription(BaseModel):
    task_name: str = Field(..., description="A friendly name for the task, e.g., 'ConvertSeattleToGPSCoords', 'MathCaclulationStep1'")
    tool_id: str = Field(..., description="Identifier of the tool responsible for executing the task")
    task_type: str = Field(..., description="Type of the task, e.g., 'fetch_data', 'process_data'")
    prompt: str = Field(..., description="The entire prompt to be sent to the agent. This should contain enough information for the agent to act. Do not use placeholders or templating")
    include_full_history: bool = Field(False, description="This should only be true when a full task history is absolutely needed. Most of the time it should be false and only inbound edge results will be included")

class TaskDependency(BaseModel):
    source: str = Field(..., description="The task name that is the source of data")
    target: str = Field(..., description="The task name that is the target of data")
    condition: Optional[dict] = Field(None, description="Conditions for the dependency to be fulfilled")
    
class TaskPlanOutput(BaseModel):
    tasks: List[TaskDescription] = None
    dependencies: Optional[List[TaskDependency]] = None

    @field_validator('tasks')
    def check_tasks(cls, tasks):
        for task in tasks:
            if not task.task_name or not task.tool_id or not task.prompt:
                raise ValueError("All tasks must have tool_id, task_name, and prompt.")
        return tasks

    @field_validator('dependencies')
    def check_dependencies(cls, dependencies):
        if dependencies is None or len(dependencies) == 0:
            return dependencies  # Return as is if dependencies is None or empty

        for dependency in dependencies:
            if not dependency.source or not dependency.target:
                raise ValueError("All dependencies must have source and target set.")
        return dependencies

class TaskNode(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task")
    task_name: str = Field(..., description="A friendly name for the task")
    graph_id: Optional[str] = Field(..., description="The graph id for the parent request this task derives from")
    tool_id: str = Field(..., description="Identifier of the agent responsible for executing the task")
    task_type: str = Field(..., description="Type of the task, e.g., 'fetch_data', 'process_data'")
    prompt: str = Field(..., description="Template for the prompt to be sent to the agent. This should contain enough information for the agent to act, as well as an {input} so additional information can be injected")
    context: Dict[str, Any] = Field(..., description="Relevant context for the task at hand")
    status: TaskStatus = Field(TaskStatus.PENDING, description="Current status of the task")
    inbound_edges: List["TaskEdge"] = Field([], description="List of incoming edges to this task node")
    outbound_edges: List["TaskEdge"] = Field([], description="List of outgoing edges from this task node")
    result: Optional[Any] = Field(None, description="Result of the task, LLM should leave this empty")
    stop_reason: Optional[str] = Field(None, description="The reason why the task was stopped, LLM should leave this empty")
    include_full_history: bool = Field(False, description="This should only be true when a full task history is absolutely needed. Most of the time it should be false and only inbound edge results will be included")
    start_time: Optional[float] = Field(None, description="The time that the task was started")
    duration: Optional[float] = Field(None, description="The number of seconds that it took to complete the task")
    llm: Optional[BaseChatModel] = Field(None, description="The LLM provider used to execute the task")

    def update_status(self, status: TaskStatus, result: Optional[str] = None):
        self.status = status
        self.result = result

    def get_child_nodes(self, task_graph: 'TaskGraphV2'):
        return [task_graph.get_node_by_task_id(edge.target_id) for edge in self.outbound_edges]

class TaskEdge(BaseModel):
    source_id: str = Field(..., description="The source task node")
    target_id: str = Field(..., description="The target task node")
    condition: str = Field(None, description="Method to check if the edge should be traversed")
    traverse_count: int = Field(0, description="Number of times the edge has been traversed")

class TaskGraphV2:
    """
    This is a clone of the TaskGraph class, as I'm looking to abstract the implementation of the task graph 
    to support either high level planning or low level planning. This should eventually replace the original
    TaskGraph class. 
    """
    nodes: Dict[str, TaskNode]
    edges: List[TaskEdge]
    task_name_map: Dict[str, str]  # Map task_name to task_id
    start_time: Optional[float] = Field(..., description="The time that the request arrived")
    history: List[str] = Field(..., description="History of the task graph calls")
    graph_id: Optional[str] = Field(None, description="Unique identifier for the task graph")
    request_id: Optional[str] = Field(None, description="Unique identifier for the request")
    state_def: Any = Field(None, description="State definition for the task graph")

    def __init__(self, task_plan: TaskPlanOutput, init_state: Any = None):
        tasks = task_plan.tasks
        dependencies = task_plan.dependencies
        
        self.nodes = {}
        self.edges = []
        self.task_name_map = {}
        self.history = list()
        
        self.start_time = time.time()
        self.graph_id = 'graph_' + str(uuid.uuid4()).split('-')[-1]
        
        if self.state_def is not None and not isinstance(init_state, type(self.state_def)):
            raise TypeError(f"init_state must be of type {type(self.state_def).__name__}")

        if tasks is None:
            # Handle the case where tasks is None
            print("No tasks provided")
            return

        # Create task nodes
        for task in tasks:
            task_id = 'task_' + str(uuid.uuid4()).split('-')[-1]  # Use only the last part of the UUID
            node = TaskNode(
                task_id=task_id,
                task_name=task.task_name,
                graph_id=self.graph_id,
                tool_id=task.tool_id,
                status=TaskStatus.PENDING,
                task_type=task.task_type,
                prompt=task.prompt,
                include_full_history=task.include_full_history
            )
            self.nodes[task_id] = node
            self.task_name_map[task.task_name] = task_id

        # Create task edges based on dependencies
        if dependencies and len(tasks) > 1:
            for dependency in dependencies:
                source_name = dependency.source
                target_name = dependency.target
                condition = dependency.condition

                source_id = self.task_name_map[source_name]
                target_id = self.task_name_map[target_name]

                source_node = self.nodes[source_id]
                target_node = self.nodes[target_id]

                edge = TaskEdge(source_id=source_node.task_id, target_id=target_node.task_id, condition=condition)

                source_node.outbound_edges.append(edge)
                target_node.inbound_edges.append(edge)
                self.edges.append(edge)

    def get_node_by_task_id(self, task_id: str) -> Optional[TaskNode]:
        return self.nodes.get(task_id)

    def get_child_nodes(self, nodes: Dict[str, TaskNode], node: TaskNode) -> List[TaskNode]:
        child_nodes = []
        for edge in node.outbound_edges:
            child_nodes.append(nodes[edge.target_id])
        return child_nodes

    def is_complete(self) -> bool:
        return all(node.status == TaskStatus.COMPLETED for node in self.nodes.values())

    def get_failed_tasks(self) -> List[TaskNode]:
        return [node for node in self.nodes.values() if node.status in [TaskStatus.FAILED, TaskStatus.RETRIESEXCEEDED]]

    def to_dict(self) -> Dict:
        nodes_data = [{"task_id": node.task_id, "task_name": node.task_name, "status": node.status.value} for node in self.nodes.values()]
        edges_data = [{"source": edge.source_id, "target": edge.target_id} for edge in self.edges]
        return {"nodes": nodes_data, "edges": edges_data}

    def get_entrypoint_nodes(self) -> List[TaskNode]:
        """
        Returns a list of all top-level leaf nodes in the task graph.
        A top-level leaf node is a node that has no inbound edges.
        """
        # Initialize an empty list to store the top-level leaf nodes
        top_level_leaf_nodes = []

        # Iterate over all nodes in the task graph
        for node in self.nodes.values():
            # Check if the node has no inbound edges
            if not node.inbound_edges:
                # If it has no inbound edges, add it to the list of top-level leaf nodes
                top_level_leaf_nodes.append(node)

        # Return the list of top-level leaf nodes
        return top_level_leaf_nodes
    
    def get_terminal_nodes(self) -> List[TaskNode]:
        """
        Returns a list of all terminal nodes in the task graph.
        A terminal node is a node that has no outbound edges.
        """
        terminal_nodes = []

        for node in self.nodes.values():
            if not node.outbound_edges:
                terminal_nodes.append(node)

        return terminal_nodes

    def is_complete(self) -> bool:
        return all(node.status == TaskStatus.COMPLETED for node in self.nodes.values())

    def get_ready_tasks(self) -> List[TaskNode]:
        ready_nodes = []
        for node in self.nodes.values():
            if node.status == TaskStatus.PENDING and all(self.nodes[edge.source_id].status == TaskStatus.COMPLETED for edge in node.inbound_edges):
                ready_nodes.append(node)
        return ready_nodes

    def mark_task_completed(self, task_id: str, result: Optional[str] = None) -> List[TaskNode]:
        node = self.get_node_by_task_id(task_id)
        if node:
            node.update_status(TaskStatus.COMPLETED, result)
            self.history.append({"task_id": node.task_id, "agent_id": node.tool_id, "status": node.status, "result": node.result})
            return self.get_ready_tasks()

    def get_task_history(self, task_id: str) -> List[str]:
        """
        Get the history of a specific task node, either a full history of all previous tasks or just the results of the parent tasks.
        Determines whether or not to provide full history by checking the `include_full_history` attribute of the task node.

        :param task_id: The task ID of the node for which to retrieve the history.
        :return: A list of strings, either the full history of the task graph or just the results of the parent tasks.
        """
        node = self.get_node_by_task_id(task_id)
        if node is None:
            raise ValueError(f"Task node with id {task_id} not found in the task graph.")

        history = []
        if node.include_full_history:
            # TODO: Make max history size configurable when including full history
            history = self.history
        else:
            for edge in node.inbound_edges:
                parent_node = self.get_node_by_task_id(edge.source_id)
                if parent_node.result is not None:
                    history.append(parent_node.result)
                    
        if len(history) == 0:
            history = [""]
            
        return history
