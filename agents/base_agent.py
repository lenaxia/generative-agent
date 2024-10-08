from logging import Logger
from abc import abstractmethod
from typing import Any, Dict, Optional
from langchain.agents import AgentType
from llm_provider.factory import LLMFactory, LLMType
from langchain.tools import BaseTool
from langchain_core.runnables.base import Runnable
from shared_tools.message_bus import MessageBus, MessageType

class BaseAgent:
    # TODO: Potentially need to rename this class to not conflict with the LangChain BaseAgent Class that I think exists
    def __init__(self, logger: Logger, llm_factory: LLMFactory, message_bus: MessageBus, agent_id: str, config: Optional[Dict] = None, agent_description: Optional[str] = None):
        self.llm_factory = llm_factory
        self.config = config or {}
        self.state = None
        self.version = None 
        self.message_bus = message_bus
        self.agent_id = agent_id
        self.agent_description = agent_description or None
        self.logger = logger

        self.subscribe_to_messages(MessageType.TASK_ASSIGNMENT, self.handle_task_assignment)

    @property
    def tools(self) -> Dict[str, BaseTool]:
        """
        Returns a dictionary of tools that the agent can use.
        """
        raise NotImplementedError

    def _run(self, llm_provider, *args, **kwargs) -> Any:
        """
        Executes the agent's task synchronously.
        """
        raise NotImplementedError

    def _arun(self, llm_provider, *args, **kwargs) -> Any:
        """
        Executes the agent's task asynchronously.
        """
        raise NotImplementedError

    def _format_input(self, *args, **kwargs) -> Any:
        """
        Formats the input for the LLM provider and the respective tool(s).
        """
        raise NotImplementedError

    def _process_output(self, *args, **kwargs) -> Any:
        """
        Processes the output from the LLM provider and the respective tool(s).
        """
        raise NotImplementedError


    def run(self, instruction: str, llm_type: LLMType = LLMType.DEFAULT, *args, **kwargs) -> Any:
        """
        Executes the agent's task synchronously.
        """
        self.setup()
        input_data = self._format_input(instruction, *args, **kwargs)

        llm_provider = self._select_llm_provider(llm_type, **kwargs)
        output_data = self._run(llm_provider, input_data)
        self.teardown()
        return self._process_output(output_data)

    def _select_llm_provider(self, llm_type: LLMType, **kwargs) -> Runnable:
        """
        Selects the LLM provider based on the specified type and additional arguments.
        Subclasses can override this method to customize LLM provider selection.
        """
        return self.llm_factory.create_provider(llm_type, **kwargs)

    async def arun(self, instruction: str, llm_type: LLMType = LLMType.DEFAULT, *args, **kwargs) -> Any:
        """
        Executes the agent's task asynchronously.
        """
        self.setup()
        llm_provider = self.llm_factory.create_provider(llm_type, **kwargs)
        input_data = self._format_input(instruction, *args, **kwargs)
        output_data = await self._arun(llm_provider, input_data)
        self.teardown()
        return self._process_output(output_data)

    def handle_task_assignment(self, task_data: Dict):
        if task_data["agent_id"] != self.agent_id:
            self.logger.info(f"Request seen by {self.agent_id} but it is not directed at me. Request ID: {task_data['request_id']}")
            return 

        self.logger.info(f"New Request Received by {self.agent_id}, Request ID: {task_data['request_id']}, Task ID: {task_data['task_id']}")

        try:
            task_id = task_data["task_id"]
            agent_id = task_data["agent_id"]
            task_type = task_data["task_type"]
            prompt = task_data["prompt"]
            request_id = task_data["request_id"]
    

            # Use the llm_factory instance to create the LLM provider
            llm_provider = self._select_llm_provider(LLMType.DEFAULT)
    
            # Use the provided llm_provider instance to respond to the task
            result = self._run(llm_provider, prompt)

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
            self.logger.error(f"Error handling task assignment for task '{task_id}': {e}")
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
