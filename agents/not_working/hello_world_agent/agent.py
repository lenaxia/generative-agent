from logging import Logger
from typing import Any, Dict
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_provider.factory import LLMFactory, LLMType
from common.message_bus import MessageBus
from agents.base_agent import BaseAgent, AgentInput
from langchain_core.messages import HumanMessage, SystemMessage

class HelloWorldAgent(BaseAgent):
    def __init__(self, logger: Logger, llm_factory: LLMFactory, message_bus: MessageBus, agent_id: str, config: Dict = None):
        super().__init__(logger, llm_factory, message_bus, agent_id, config)
        self.prompt_template = ChatPromptTemplate.from_template("{instruction}")
        self.output_parser = StrOutputParser()
        self.llm_factory = llm_factory
        self.agent_description = "Just say hello to the world"
        
    @property
    def tools(self) -> Dict[str, BaseTool]:
        return {}

    def _run(self, input: AgentInput) -> Any:
        llm_provider = self._select_llm_provider(LLMType.DEFAULT)
        
        messages = [
            SystemMessage(
                content="You are a HelloWorld bot which responds with Hello World in various languages to all requests."
            ),
            HumanMessage(
                content=input.prompt,
            )
        ]
        return llm_provider.invoke(messages)

    def _arun(self, llm_provider, instruction: str) -> Any:
        raise NotImplementedError("Asynchronous execution not supported.")

    def _format_input(self, instruction: str, *args, **kwargs) -> AgentInput:
        return AgentInput(prompt=instruction)

    def _process_output(self, output: str, *args, **kwargs) -> str:
        return output

    def setup(self):
        pass

    def teardown(self):
        pass

