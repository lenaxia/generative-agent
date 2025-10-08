from logging import Logger
from abc import abstractmethod
from typing import Any, Dict, Optional, List
from anthropic import BaseModel
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.output_parsers import PydanticOutputParser

from common.task_graph import TaskPlanOutput
from agents.base_agent import AgentInput, BaseAgent
from llm_provider.factory import LLMFactory, LLMType
from common.message_bus import MessageBus, MessageType
import agents.coding_agent.tools.core as CoreTools

class CodingAgentConfig(BaseModel):
    llm_class: str = "default"
    work_dir: str = None


    slack_channel: str
    status_channel: str
    monitored_event_types: List[str] = ["message"]
    online_message: str = "SlackAgent is online and ready to receive messages."
    llm_class: str = "default"
    history_limit: int = 5


class CodingAgent(BaseAgent):
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
        return {
            #CoreTools.draft_code,
            CoreTools.retrieve_context,
            CoreTools.use_aider
            }

    def _create_plan(self, input: AgentInput) -> Any:
        """Creates a plan for completing the given task.
        """

        coding_parser = PydanticOutputParser(pydantic_object=TaskPlanOutput)

        prompt_template = PromptTemplate(
            input_variables=["input", "tools"],
            template="You are a planning agent responsible for breaking down complex tasks into a sequence of smaller subtasks."
                     "Given the available tools and their capabilities, create a task graph to accomplish the following task: {input}\n\n"
                     "These are the tools available to you:\n{tools}\n\n"
                     "Your output should follow the provided JSON schema:\n\n{format_instructions}\n\n"
                     "Only respond with the JSON and no additional formatting or comments, do not include anything else besides the json output\n"
                     "Task Graph:\n",
            partial_variables={"format_instructions": coding_parser.get_format_instructions()},
        )

        llm = self._select_llm_provider()
        tools = [tool for tool in self.tools]

        tools_prompt =  "\n".join([f"- {tool.model_fields['name'].default}: {tool.model_fields['description'].default}" for tool in tools])

        chain = prompt_template | llm | coding_parser

        planning_agent_output = chain.invoke({"input": input.prompt, "tools": tools_prompt})

        return planning_agent_output

    def _run(self, input: AgentInput) -> Any:
        """Executes the agent's task synchronously.
        """
        plan = self._create_plan(input)

        return plan

    def _select_llm_provider(self) -> Runnable:
        """Selects the LLM provider based on the specified type and additional arguments.
        Subclasses can override this method to customize LLM provider selection.
        """
        return self.llm_factory.create_chat_model(LLMType.DEFAULT)

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
