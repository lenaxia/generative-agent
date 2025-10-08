from logging import Logger
from abc import abstractmethod
from typing import Any, Dict, Optional, List, Union, Callable
from common.bus_packet import BusPacket
from llm_provider.factory import LLMFactory, LLMType
from langchain.tools import BaseTool
from langchain_core.runnables.base import Runnable
from common.message_bus import MessageBus, MessageType
from pydantic import BaseModel, Field

class AgentInput(BaseModel):
    prompt: str
    history: Optional[Union[List[Any], None]] = None
    llm_type: Optional[LLMType] = LLMType.DEFAULT
    additional_data: Optional[Dict[str, Any]] = Field(default_factory=dict)

class BaseAgent:
    def __init__(self, logger: Logger, llm_factory: LLMFactory, message_bus: MessageBus, agent_id: str, config: Optional[Dict] = None, agent_description: Optional[str] = None):
        self.llm_factory = llm_factory
        self.config = config or {}
        self.state = None
        self.version = None
        self.message_bus = message_bus
        self.agent_id = agent_id
        self.agent_type = None
        self.agent_description = agent_description or None
        self.logger = logger

        self.subscribe_to_messages(MessageType.TASK_ASSIGNMENT, self.handle_task_assignment)

    @property
    def tools(self) -> Dict[str, BaseTool]:
        """Returns a dictionary of tools that the agent can use.
        """
        raise NotImplementedError

    def _run(self, input: AgentInput) -> Any:
        """Executes the agent's task synchronously.
        """
        raise NotImplementedError

    def _arun(self, input: AgentInput) -> Any:
        """Executes the agent's task asynchronously.
        """
        raise NotImplementedError

    def _format_input(self, task_data: Dict) -> AgentInput:
        """Formats the input for the LLM provider and the respective tool(s).
        """
        input: AgentInput = AgentInput(**task_data)
        return input

    def _process_output(self, task_data: Dict, output: Any) -> Any:
        """Processes the output from the LLM provider and the respective tool(s).
        """
        return output

    def run(self, task_data: Dict) -> Any:
        """Executes the agent's task synchronously.
        """
        self.setup()
        input_data: AgentInput = self._format_input(task_data)
        output_data = self._run(input_data)
        self.teardown()
        return self._process_output(task_data, output_data)

    def _select_llm_provider(self) -> Runnable:
        """Selects the LLM provider based on the specified type and additional arguments.
        Subclasses can override this method to customize LLM provider selection.
        """
        return self.llm_factory.create_chat_model(LLMType.DEFAULT)

    async def arun(self, input: AgentInput) -> Any:
        """Executes the agent's task asynchronously.
        """
        self.setup()
        llm: Runnable = self.llm_factory.create_chat_model(input.llm_type)
        input_data: AgentInput = self._format_input(input.prompt)
        output_data = await self._arun(llm, input_data)
        self.teardown()
        return self._process_output(input, output_data)

    def handle_task_assignment(self, task_data: Dict):
        if task_data["agent_id"] != self.agent_id:
            self.logger.debug(f"Request seen by {self.agent_id} but it is not directed at me. Request ID: {task_data['request_id']}, Task ID: {task_data['task_id']}")
            return

        self.logger.info(f"New Request Received by {self.agent_id}, Request ID: {task_data['request_id']}, Task ID: {task_data['task_id']}")

        try:
            task_id = task_data["task_id"]
            request_id = task_data["request_id"]

            # Use the provided llm_provider instance to respond to the task
            result = self.run(task_data)

            # Publish the task response on the MessageBus
            task_data["result"] = result

            self.message_bus.publish(self, MessageType.TASK_RESPONSE, task_data)
        except Exception as e:
            self.logger.error(f"Error handling task assignment for task '{task_id}': {e}")
            # Publish an error message on the MessageBus
            error_data = {
                "request_id": request_id,
                "task_id": task_id,
                "error_message": str(e),
            }
            self.message_bus.publish(self, MessageType.AGENT_ERROR, error_data)

    @abstractmethod
    def initialize(self):
        """Performs initialization operations
        """

    @abstractmethod
    def setup(self):
        """
        Performs setup operations before task execution.
        """

    @abstractmethod
    def teardown(self):
        """Performs teardown operations after task execution.
        """

    def persist_state(self):
        """Persists the agent's state for future use or recovery.
        """
        # Implement state persistence logic here

    def load_state(self):
        """Loads the agent's state from persistence storage.
        """
        # Implement state loading logic here

    def upgrade(self, new_version):
        """Upgrades the agent to a new version.
        """
        # Implement agent upgrade logic here

    def publish_message(self, message_type, message):
        """Publishes a message to the MessageBus for other agents to consume.
        """
        self.message_bus.publish(self, message_type, message)

    def publish_packet(self, packet: BusPacket):
        """Publishes a message to the MessageBus for other agents to consume.
        """
        self.message_bus.publish(self, packet)

    def subscribe_to_messages(self, message_type, callback):
        """Subscribes to messages of a specific type from the MessageBus.
        """
        self.message_bus.subscribe(self, message_type, callback)

    def create_tool_from_agent(self) -> Callable:
        """Converts the current agent into a tool, relying on a pydantic to validate input.
        This is meant to allow this agent to be called as a tool as part of a high level
        supervisor agent/workflow.
        """
        def tool_func(input_data: Dict[str, Any]) -> Any:
            input_model_instance = AgentInput(**input_data)
            return self.run(input_model_instance.prompt, history=input_model_instance.history, llm_type=input_model_instance.llm_type, **input_model_instance.additional_data)
        return tool_func
