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
    prompt: str = Field(..., description="The entire prompt to be sent to the agent. This should contain enough information for the agent to act. Do not use placeholders or templating")
    include_full_history: bool = Field(False, description="This should only be true when a full task history is absolutely needed. Most of the time it should be false and only inbound edge results will be included")

class TaskDependency(BaseModel):
    source: str = Field(..., description="The task name that is the source of data")
    target: str = Field(..., description="The task name that is the target of data")
    condition: Optional[dict] = Field(None, description="Conditions for the dependency to be fulfilled")

class TaskNode(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task")
    task_name: str = Field(..., description="A friendly name for the task")
    request_id: Optional[str] = Field(..., description="The request id for the parent request this task derives from")
    agent_id: str = Field(..., description="Identifier of the agent responsible for executing the task")
    agent_config: dict = Field(default_factory=dict, description="Configuration for the agent responsible for executing the task")
    task_type: str = Field(..., description="Type of the task, e.g., 'fetch_data', 'process_data'")
    prompt: str = Field(..., description="Template for the prompt to be sent to the agent. This should contain enough information for the agent to act, as well as an {input} so additional information can be injected")
    status: TaskStatus = Field(TaskStatus.PENDING, description="Current status of the task")
    inbound_edges: List["TaskEdge"] = Field([], description="List of incoming edges to this task node")
    outbound_edges: List["TaskEdge"] = Field([], description="List of outgoing edges from this task node")
    result: Optional[str] = Field(None, description="Result of the task, LLM should leave this empty")
    stop_reason: Optional[str] = Field(None, description="The reason why the task was stopped, LLM should leave this empty")
    include_full_history: bool = Field(False, description="This should only be true when a full task history is absolutely needed. Most of the time it should be false and only inbound edge results will be included")
    start_time: Optional[float] = Field(None, description="The time that the task was started")
    duration: Optional[float] = Field(None, description="The number of seconds that it took to complete the task")
    
    # New fields for StrandsAgent integration
    role: Optional[str] = Field(None, description="Role for Universal Agent execution (e.g., 'planning', 'search', 'summarizer')")
    llm_type: Optional[str] = Field(None, description="LLM type for execution (e.g., 'STRONG', 'WEAK', 'DEFAULT')")
    required_tools: List[str] = Field(default_factory=list, description="List of tools required for this task")
    task_context: dict = Field(default_factory=dict, description="Context data for task execution")

    def update_status(self, status: TaskStatus, result: Optional[str] = None):
        self.status = status
        self.result = result

    def get_child_nodes(self, task_graph: 'TaskGraph'):
        return [task_graph.get_node_by_task_id(edge.target_id) for edge in self.outbound_edges]

    # StrandsAgent integration methods
    def set_role(self, role: str):
        """Set the role for Universal Agent execution."""
        self.role = role

    def get_role(self) -> Optional[str]:
        """Get the role for Universal Agent execution."""
        return self.role

    def set_llm_type(self, llm_type: str):
        """Set the LLM type for execution."""
        self.llm_type = llm_type

    def get_llm_type(self) -> Optional[str]:
        """Get the LLM type for execution."""
        return self.llm_type

    def set_required_tools(self, tools: List[str]):
        """Set the required tools for this task."""
        self.required_tools = tools.copy()

    def get_required_tools(self) -> List[str]:
        """Get the required tools for this task."""
        return self.required_tools.copy()

    def set_context(self, context: dict):
        """Set context data for task execution."""
        self.task_context = context.copy()

    def get_context(self) -> dict:
        """Get context data for task execution."""
        return self.task_context.copy()

class TaskEdge(BaseModel):
    source_id: str = Field(..., description="The source task node")
    target_id: str = Field(..., description="The target task node")
    condition: Optional[dict] = Field(None, description="Conditions for the edge to be traversed")

class TaskGraph:
    nodes: Dict[str, TaskNode]
    edges: List[TaskEdge]
    task_name_map: Dict[str, str]  # Map task_name to task_id
    start_time: Optional[float] = Field(..., description="The time that the request arrived")
    history: List[str] = Field(..., description="History of the task graph calls")
    graph_id: Optional[str] = Field(None, description="Unique identifier for the task graph")
    request_id: Optional[str] = Field(None, description="Unique identifier for the request")
    
    # New fields for external state management
    conversation_history: List[Dict] = Field(default_factory=list, description="Conversation history")
    progressive_summary: List[str] = Field(default_factory=list, description="Progressive summary of task execution")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata for the task graph")

    def __init__(self, tasks: List[TaskDescription], dependencies: Optional[List[Dict]] = None, request_id: Optional[str] = ""):
        self.nodes = {}
        self.edges = []
        self.task_name_map = {}
        self.start_time = time.time()
        self.request_id = request_id
        self.history = list()
        self.graph_id = 'graph_' + str(uuid.uuid4()).split('-')[-1]
        
        # Initialize new fields for external state management
        self.conversation_history = []
        self.progressive_summary = []
        self.metadata = {}

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
                request_id=self.request_id,
                agent_id=task.agent_id,
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
            self.history.append({"task_id": node.task_id, "agent_id": node.agent_id, "status": node.status, "result": node.result})
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
            history = ["The beginning"]
            
        return history

    def create_checkpoint(self) -> Dict:
        """
        Create a checkpoint of the current task graph state.
        
        Returns:
            Dict: Checkpoint containing task graph state, conversation history,
                  progressive summary, and metadata
        """
        checkpoint = {
            'timestamp': time.time(),
            'task_graph_state': {
                'nodes': [
                    {
                        'task_id': node.task_id,
                        'task_name': node.task_name,
                        'status': node.status.value,
                        'result': node.result,
                        'agent_id': node.agent_id,
                        'task_type': node.task_type,
                        'prompt': node.prompt,
                        'start_time': node.start_time,
                        'duration': node.duration,
                        'include_full_history': node.include_full_history,
                        'role': node.role,
                        'llm_type': node.llm_type,
                        'required_tools': node.required_tools,
                        'task_context': node.task_context
                    } for node in self.nodes.values()
                ],
                'edges': [
                    {
                        'source_id': edge.source_id,
                        'target_id': edge.target_id,
                        'condition': edge.condition
                    } for edge in self.edges
                ],
                'task_name_map': self.task_name_map.copy(),
                'history': self.history.copy(),
                'graph_id': self.graph_id,
                'request_id': self.request_id,
                'start_time': self.start_time
            },
            'conversation_history': self.conversation_history.copy(),
            'progressive_summary': self.progressive_summary.copy(),
            'metadata': self.metadata.copy()
        }
        return checkpoint

    def resume_from_checkpoint(self, checkpoint: Dict):
        """
        Resume task graph execution from a checkpoint.
        
        Args:
            checkpoint (Dict): Checkpoint data containing task graph state
            
        Raises:
            ValueError: If checkpoint data is invalid
        """
        if not isinstance(checkpoint, dict) or 'task_graph_state' not in checkpoint:
            raise ValueError("Invalid checkpoint data: missing task_graph_state")
            
        try:
            task_graph_state = checkpoint['task_graph_state']
            
            # Restore basic properties
            self.graph_id = task_graph_state.get('graph_id', self.graph_id)
            self.request_id = task_graph_state.get('request_id', self.request_id)
            self.start_time = task_graph_state.get('start_time', self.start_time)
            self.history = task_graph_state.get('history', []).copy()
            self.task_name_map = task_graph_state.get('task_name_map', {}).copy()
            
            # Restore nodes
            self.nodes = {}
            for node_data in task_graph_state.get('nodes', []):
                node = TaskNode(
                    task_id=node_data['task_id'],
                    task_name=node_data['task_name'],
                    request_id=node_data.get('request_id', self.request_id),
                    agent_id=node_data['agent_id'],
                    task_type=node_data['task_type'],
                    prompt=node_data['prompt'],
                    status=TaskStatus(node_data['status']),
                    result=node_data.get('result'),
                    start_time=node_data.get('start_time'),
                    duration=node_data.get('duration'),
                    include_full_history=node_data.get('include_full_history', False),
                    role=node_data.get('role'),
                    llm_type=node_data.get('llm_type'),
                    required_tools=node_data.get('required_tools', []),
                    task_context=node_data.get('task_context', {})
                )
                self.nodes[node.task_id] = node
            
            # Restore edges
            self.edges = []
            for edge_data in task_graph_state.get('edges', []):
                edge = TaskEdge(
                    source_id=edge_data['source_id'],
                    target_id=edge_data['target_id'],
                    condition=edge_data.get('condition')
                )
                self.edges.append(edge)
                
                # Restore edge references in nodes
                source_node = self.nodes.get(edge.source_id)
                target_node = self.nodes.get(edge.target_id)
                if source_node and target_node:
                    source_node.outbound_edges.append(edge)
                    target_node.inbound_edges.append(edge)
            
            # Restore external state
            self.conversation_history = checkpoint.get('conversation_history', []).copy()
            self.progressive_summary = checkpoint.get('progressive_summary', []).copy()
            self.metadata = checkpoint.get('metadata', {}).copy()
            
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid checkpoint data: {str(e)}")

    def add_conversation_entry(self, role: str, content: str):
        """
        Add an entry to the conversation history.
        
        Args:
            role (str): Role of the speaker (e.g., 'user', 'assistant')
            content (str): Content of the message
        """
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': time.time()
        })

    def add_to_progressive_summary(self, summary: str):
        """
        Add to the progressive summary.
        
        Args:
            summary (str): Summary text to add
        """
        self.progressive_summary.append({
            'summary': summary,
            'timestamp': time.time()
        })

    def set_metadata(self, key: str, value):
        """
        Set metadata for the task graph.
        
        Args:
            key (str): Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default=None):
        """
        Get metadata value.
        
        Args:
            key (str): Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)

    def get_task_context(self, task_id: str) -> dict:
        """
        Get comprehensive context for a task including parent results,
        conversation history, and progressive summary.
        
        Args:
            task_id (str): Task ID to get context for
            
        Returns:
            dict: Context containing parent results, conversation history, etc.
        """
        node = self.get_node_by_task_id(task_id)
        if not node:
            raise ValueError(f"Task node with id {task_id} not found")
        
        # Get parent task results
        parent_results = []
        for edge in node.inbound_edges:
            parent_node = self.get_node_by_task_id(edge.source_id)
            if parent_node and parent_node.result:
                try:
                    # Try to parse as JSON, fall back to string
                    import json
                    result_data = json.loads(parent_node.result)
                except (json.JSONDecodeError, TypeError):
                    result_data = parent_node.result
                parent_results.append(result_data)
        
        context = {
            "task_id": task_id,
            "task_name": node.task_name,
            "parent_results": parent_results,
            "conversation_history": self.conversation_history.copy(),
            "progressive_summary": self.progressive_summary.copy(),
            "metadata": self.metadata.copy(),
            "task_context": node.get_context()
        }
        
        return context

    def prepare_task_execution(self, task_id: str) -> dict:
        """
        Prepare configuration for task execution with Universal Agent.
        
        Args:
            task_id (str): Task ID to prepare execution for
            
        Returns:
            dict: Execution configuration including role, tools, context, etc.
        """
        node = self.get_node_by_task_id(task_id)
        if not node:
            raise ValueError(f"Task node with id {task_id} not found")
        
        context = self.get_task_context(task_id)
        
        execution_config = {
            "role": node.get_role() or "default",
            "llm_type": node.get_llm_type() or "DEFAULT",
            "tools": node.get_required_tools(),
            "prompt": node.prompt,
            "context": context,
            "task_id": task_id,
            "task_name": node.task_name,
            "task_type": node.task_type
        }
        
        return execution_config

    def mark_task_completed(self, task_id: str, result: Optional[str] = None) -> List[TaskNode]:
        """
        Enhanced version that also updates progressive summary.
        """
        node = self.get_node_by_task_id(task_id)
        if node:
            node.update_status(TaskStatus.COMPLETED, result)
            self.history.append({
                "task_id": node.task_id,
                "agent_id": node.agent_id,
                "status": node.status,
                "result": node.result
            })
            
            # Auto-update progressive summary
            summary_text = f"Completed {node.task_name} ({node.task_type})"
            if result:
                # Truncate long results for summary
                result_preview = result[:100] + "..." if len(result) > 100 else result
                summary_text += f": {result_preview}"
            
            self.add_to_progressive_summary(summary_text)
            
            return self.get_ready_tasks()
        return []
