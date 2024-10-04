from logging import Logger
from abc import abstractmethod
from typing import Any, Dict, Optional
from langchain.agents import AgentType
from llm_provider.base_client import BaseLLMClient
from langchain.tools import BaseTool
from supervisor.llm_registry import LLMRegistry, LLMType
from shared_tools.message_bus import MessageBus

class BaseAgent:
    def __init__(self, logger: Logger, llm_registry: LLMRegistry, message_bus: MessageBus, agent_id: str, config: Optional[Dict] = None):
        self.llm_registry = llm_registry
        self.config = config or {}
        self.state = None
        self.version = None
        self.message_bus = None
        self.agent_id = agent_id
        self.logger = logger

    @property
    def tools(self) -> Dict[str, BaseTool]:
        """
        Returns a dictionary of tools that the agent can use.
        """
        raise NotImplementedError

    def _run(self, llm_type: LLMType = LLMType.DEFAULT, *args, **kwargs) -> Any:
        """
        Executes the agent's task synchronously.
        """
        raise NotImplementedError

    def _arun(self, llm_type: LLMType = LLMType.DEFAULT, *args, **kwargs) -> Any:
        """
        Executes the agent's task asynchronously.
        """
        raise NotImplementedError

    def _format_input(self, *args, **kwargs) -> Any:
        """
        Formats the input for the LLM client and the respective tool(s).
        """
        raise NotImplementedError

    def _process_output(self, *args, **kwargs) -> Any:
        """
        Processes the output from the LLM client and the respective tool(s).
        """
        raise NotImplementedError

    def run(self, instruction: str, llm_type: LLMType = LLMType.DEFAULT, *args, **kwargs) -> Any:
        """
        Executes the agent's task synchronously.
        """
        self.setup()
        llm_client = self.llm_registry.get_client(llm_type)
        input_data = self._format_input(instruction, *args, **kwargs)
        output_data = self._run(llm_client, input_data)
        self.teardown()
        return self._process_output(output_data)

    async def arun(self, instruction: str, llm_type: LLMType = LLMType.DEFAULT, *args, **kwargs) -> Any:
        """
        Executes the agent's task asynchronously.
        """
        self.setup()
        llm_client = self.llm_registry.get_client(llm_type)
        input_data = self._format_input(instruction, *args, **kwargs)
        output_data = await self._arun(llm_client, input_data)
        self.teardown()
        return self._process_output(output_data)

    def set_message_bus(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.message_bus.subscribe(self, MessageType.TASK_ASSIGNMENT, self.handle_task_assignment)
        
    def handle_task_assignment(self, task_data: Dict):
        try:
            task_id = task_data["task_id"]
            agent_id = task_data["agent_id"]
            task_type = task_data["task_type"]
            prompt = task_data["prompt"]
            request_id = task_data["request_id"]
            llm_client: BaseLLMClient = task_data["llm_client"]

            # Use the provided llm_client instance to respond to the task
            result = self._run(llm_client, prompt)

            # Publish the task response on the MessageBus
            response_data = {
                "task_id": task_id,
                "agent_id": agent_id,
                "task_type": task_type,
                "result": result,
                "request_id": request_id,
            }
            self.message_bus.publish(self, MessageType.TASK_RESPONSE, response_data)
        except Exception as e:
            logger.error(f"Error handling task assignment for task '{task_id}': {e}")
            # Publish an error message on the MessageBus
            error_data = {
                "request_id": request_id,
                "task_id": task_id,
                "error_message": str(e),
            }
            self.message_bus.publish(self, MessageType.AGENT_ERROR, error_data)

    @abstractmethod
    def setup(self):
        """
        Performs setup operations before task execution.
        """
        pass

    @abstractmethod
    def teardown(self):
        """
        Performs teardown operations after task execution.
        """
        pass

    def persist_state(self):
        """
        Persists the agent's state for future use or recovery.
        """
        # Implement state persistence logic here
        pass

    def load_state(self):
        """
        Loads the agent's state from persistence storage.
        """
        # Implement state loading logic here
        pass

    def upgrade(self, new_version):
        """
        Upgrades the agent to a new version.
        """
        # Implement agent upgrade logic here
        pass

    def publish_message(self, message_type, message):
        """
        Publishes a message to the MessageBus for other agents to consume.
        """
        self.message_bus.publish(self, message_type, message)

    def subscribe_to_messages(self, message_type, callback):
        """
        Subscribes to messages of a specific type from the MessageBus.
        """
        self.message_bus.subscribe(self, message_type, callback)
