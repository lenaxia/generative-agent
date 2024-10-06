from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import uuid

from pydantic import BaseModel

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TaskDescription(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task")
    agent_id: str = Field(..., description="Identifier of the agent responsible for executing the task")
    task_type: str = Field(..., description="Type of the task, e.g., 'fetch_data', 'process_data'")
    prompt_template: str = Field(..., description="Template for the prompt to be sent to the agent. This should contain enough information for the agent to act, as well as an {input} so additional information can be injected")
    prompt_args: dict = Field(None, description="Arguments to be used in the prompt template")

class TaskDependency(BaseModel):
    source: str = Field(..., description="The task ID that is the source of data")
    target: str = Field(..., description="The task ID that is the target of data")
    condition: dict = Field(None, description="Conditions for the dependency to be fulfilled")

class TaskNode(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task")
    agent_id: str = Field(..., description="Identifier of the agent responsible for executing the task")
    task_type: str = Field(..., description="Type of the task, e.g., 'fetch_data', 'process_data'")
    prompt_template: str = Field(..., description="TTemplate for the prompt to be sent to the agent. This should contain enough information for the agent to act, as well as an {input} so additional information can be injected")
    prompt_args: Dict = Field({}, description="Arguments to be used in the prompt template")
    status: TaskStatus = Field(TaskStatus.PENDING, description="Current status of the task")
    inbound_edges: List["TaskEdge"] = Field([], description="List of incoming edges to this task node")
    outbound_edges: List["TaskEdge"] = Field([], description="List of outgoing edges from this task node")

class TaskEdge(BaseModel):
    source: TaskNode = Field(..., description="The source task node")
    target: TaskNode = Field(..., description="The target task node")
    condition: Optional[Dict] = Field(None, description="Conditions for the edge to be traversed")

class TaskGraph:
    nodes: Dict[str, TaskNode]
    edges: List[TaskEdge]

    def __init__(self, tasks: List["TaskDescription"], dependencies: Optional[List[Dict]] = None):
        self.nodes = {}
        self.edges = []

        print("BLAH123")
        print(tasks)
        print(dependencies)

        # Create task nodes
        for task in tasks:
            node = TaskNode(
                id=str(uuid.uuid4()),
                agent_id=task.agent_id,
                task_type=task.task_type,
                prompt_template=task.prompt_template,
                prompt_args=task.prompt_args or {},
            )
            self.nodes[node.id] = node

        # Create task edges based on dependencies
        if dependencies:
            for dependency in dependencies:
                source_id = dependency["source"]
                target_id = dependency["target"]
                condition = dependency.get("condition")

                source_node = self.nodes[source_id]
                target_node = self.nodes[target_id]

                edge = TaskEdge(source_node, target_node, condition)
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
        nodes_data = [{"id": node.id, "status": node.status.value} for node in self.nodes.values()]
        edges_data = [{"source": edge.source.id, "target": edge.target.id} for edge in self.edges]
        return {"nodes": nodes_data, "edges": edges_data}

