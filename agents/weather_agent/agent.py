from datetime import datetime
from typing import Any, Dict
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from agents.base_agent import BaseAgent
from llm_provider.factory import LLMFactory, LLMType
from shared_tools.message_bus import MessageBus

class WeatherTool(BaseTool):
    name: str = "check_weather"
    description: str = "Return the weather forecast for the specified location."

    def _run(self, location: str, at_time: datetime | None = None) -> str:
        return f"It's always sunny in {location}"

    def _arun(self, location: str, at_time: datetime | None = None) -> str:
        raise NotImplementedError("Asynchronous execution not supported.")

class WeatherAgent(BaseAgent):
    def __init__(self, logger, llm_factory: LLMFactory, message_bus: MessageBus, agent_id: str, config: Dict = None):
        super().__init__(logger, llm_factory, message_bus, agent_id, config)
        self.prompt_template = ChatPromptTemplate.from_template("{instruction}")

    @property
    def tools(self):
        return [WeatherTool()]

    def _run(self, llm_provider, instruction: str) -> Any:
        graph = create_react_agent(llm_provider, tools=self.tools)
        inputs = {"messages": [("user", instruction)]}
        message = graph.invoke(inputs)
        return message

    def _arun(self, llm_provider, instruction: str) -> Any:
        raise NotImplementedError("Asynchronous execution not supported.")

    def _format_input(self, instruction: str, *args, **kwargs) -> str:
        return instruction

    def _process_output(self, output: str, *args, **kwargs) -> str:
        return output

    def setup(self):
        pass

    def teardown(self):
        pass
