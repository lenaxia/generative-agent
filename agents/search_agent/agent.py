import requests
from datetime import datetime
from typing import Any, Dict, List
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from agents.base_agent import BaseAgent, AgentInput
from llm_provider.factory import LLMFactory, LLMType
from shared_tools.message_bus import MessageBus
from langchain_community.tools.tavily_search import TavilySearchResults

class SearchAgent(BaseAgent):
    def __init__(self, logger, llm_factory: LLMFactory, message_bus: MessageBus, agent_id: str, config: Dict = None):
        super().__init__(logger, llm_factory, message_bus, agent_id, config)
        self.logger = logger
        self.search_tool = TavilySearchResults(max_results=4)
        self.agent_description = "An agent which can search the web for arbitrary information and return a summary. If there are more specialized agents available, prefer using those."

    @property
    def tools(self):
        return [self.search_tool]

    def _select_llm_provider(self, llm_type: LLMType, **kwargs):
        llm = self.llm_factory.create_chat_model(llm_type)
        return llm

    def _run(self, input: AgentInput) -> Any:
        system_prompt = "You are a search bot who can search the internet for information to answer user queries."
        history_entries = [f"* {entry}" for entry in input.history]
        history_prompt = "This is the conversation so far:\n\n" + "\n".join(history_entries)

        llm_provider = self._select_llm_provider(LLMType.DEFAULT)
        config = {"configurable": {"thread_id": "abc123"}}
        graph = create_react_agent(llm_provider, tools=self.tools)
        inputs = {"messages": [("system", system_prompt), ("user", history_prompt), ("user", input.prompt)]}
        output = None
        for chunk in graph.stream(inputs, config):
            output = chunk
            self.logger.info(chunk)
            
        return output

    def _arun(self, llm_provider, instruction: str) -> Any:
        raise NotImplementedError("Asynchronous execution not supported.")

    def _process_output(self, task_data: Dict, output: Dict) -> str:
        result = output["agent"]["messages"][0].content
        return result

    def setup(self):
        pass

    def teardown(self):
        pass