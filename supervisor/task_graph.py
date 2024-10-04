from typing import Dict, List, Optional
from enum import Enum
import uuid

from supervisor.task_models import Task, TaskStatus
from langchain.tools import BaseTool
from pydantic import BaseModel

class TaskNode:
    def __init__(self, task: Task):
        self.task = task
        self.inbound_edges = []
        self.outbound_edges = []

class TaskEdge:
    def __init__(self, source: TaskNode, target: TaskNode, condition: Optional[Dict] = None):
        self.source = source
        self.target = target
        self.condition = condition

class TaskGraph(BaseModel):
    def __init__(self, tasks: List[Task], dependencies: Optional[List[Dict]] = None):
        self.nodes: Dict[str, TaskNode] = {}
        self.edges: List[TaskEdge] = []

        # Create task nodes
        for task in tasks:
            node = TaskNode(task)
            self.nodes[task.id] = node

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
        return all(node.task.status == TaskStatus.COMPLETED for node in self.nodes.values())

    def to_dict(self) -> Dict:
        nodes_data = [{"id": node.task.id, "status": node.task.status.value} for node in self.nodes.values()]
        edges_data = [{"source": edge.source.task.id, "target": edge.target.task.id} for edge in self.edges]
        return {"nodes": nodes_data, "edges": edges_data}
