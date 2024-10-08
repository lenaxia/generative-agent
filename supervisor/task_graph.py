from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
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
    agent_id: str = Field(..., description="Identifier of the agent responsible for executing the task")
    task_type: str = Field(..., description="Type of the task, e.g., 'fetch_data', 'process_data'")
    prompt_template: str = Field(..., description="The entire prompt to be sent to the agent. This should contain enough information for the agent to act. Do not use placeholders or templating")
    prompt_args: Optional[dict] = Field(None, description="Arguments to be used in the prompt template", exclude=True)

class TaskDependency(BaseModel):
    source: str = Field(..., description="The task name that is the source of data")
    target: str = Field(..., description="The task name that is the target of data")
    condition: Optional[dict] = Field(None, description="Conditions for the dependency to be fulfilled")

from typing import Optional

class TaskNode(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task")
    task_name: str = Field(..., description="A friendly name for the task")
    agent_id: str = Field(..., description="Identifier of the agent responsible for executing the task")
    task_type: str = Field(..., description="Type of the task, e.g., 'fetch_data', 'process_data'")
    prompt_template: str = Field(..., description="Template for the prompt to be sent to the agent. This should contain enough information for the agent to act, as well as an {input} so additional information can be injected")
    prompt_args: Optional[dict] = Field({}, description="Arguments to be used in the prompt template")
    prompt_template_formatted: Optional[str] = Field(..., description="Formatted prompt template, LLM should leave this empty")
    status: TaskStatus = Field(TaskStatus.PENDING, description="Current status of the task")
    inbound_edges: List["TaskEdge"] = Field([], description="List of incoming edges to this task node")
    outbound_edges: List["TaskEdge"] = Field([], description="List of outgoing edges from this task node")
    result: Optional[str] = Field(None, description="Result of the task, LLM should leave this empty")
    stop_reason: Optional[str] = Field(None, description="The reason why the task was stopped, LLM should leave this empty")

class TaskEdge(BaseModel):
    source: TaskNode = Field(..., description="The source task node")
    target: TaskNode = Field(..., description="The target task node")
    condition: Optional[dict] = Field(None, description="Conditions for the edge to be traversed")

class TaskGraph:
    nodes: Dict[str, TaskNode]
    edges: List[TaskEdge]
    task_name_map: Dict[str, str]  # Map task_name to task_id
    start_time: Optional[float] = Field(..., description="The time that the request arrived", exclude=True)

    def __init__(self, tasks: List[TaskDescription], dependencies: Optional[List[Dict]] = None):
        self.nodes = {}
        self.edges = []
        self.task_name_map = {}
        self.start_time = time.time()

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
                agent_id=task.agent_id,
                task_type=task.task_type,
                prompt_template=task.prompt_template,
                prompt_args=task.prompt_args or {},
                prompt_template_formatted=self._format_prompt_template(task),
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

                edge = TaskEdge(source=source_node, target=target_node, condition=condition)

                source_node.outbound_edges.append(edge)
                target_node.inbound_edges.append(edge)
                self.edges.append(edge)

    def get_node_by_task_id(self, task_id: str) -> Optional[TaskNode]:
        return self.nodes.get(task_id)

    def get_child_nodes(self, node: TaskNode) -> List[TaskNode]:
        child_nodes = []
        for edge in node.outbound_edges:
            child_nodes.append(edge.target)
        return child_nodes

    def is_complete(self) -> bool:
        return all(node.status == TaskStatus.COMPLETED for node in self.nodes.values())

    def to_dict(self) -> Dict:
        nodes_data = [{"task_id": node.task_id, "task_name": node.task_name, "status": node.status.value} for node in self.nodes.values()]
        edges_data = [{"source": edge.source.task_id, "target": edge.target.task_id} for edge in self.edges]
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
    
    def _format_prompt_template(self, task: TaskDescription) -> str:
        try:
            if task.prompt_args is None:
                return task.prompt_template
            return task.prompt_template.format(**task.prompt_args)
        except KeyError as e:
            logger.warning(f"Missing variable in prompt template for task {task.task_name}: {e}")
            return task.prompt_template
        except Exception as e:
            logger.warning(f"Error formatting prompt template for task {task.task_name}: {e}")
            return task.prompt_template
