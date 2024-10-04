from logging import Logger
from typing import Any, Dict, List
from langchain.agents import AgentType
from langchain.tools import BaseTool
from pydantic import BaseModel
from llm_provider.base_client import BaseLLMClient
from supervisor.llm_registry import LLMRegistry, LLMType
from agents.base_agent import BaseAgent
from shared_tools.message_bus import MessageBus
from .tools.web_search_tool import WebSearchTool, WebSearchInput, WebSearchOutput

class WebSearchAgent(BaseAgent):
    def __init__(self, logger: Logger, llm_registry: LLMRegistry, message_bus: MessageBus, agent_id: str, config: Dict = None):
        super().__init__(logger, llm_registry, message_bus, agent_id, config)
        self.tools = {"web_search": WebSearchTool(llm_client=self.llm_registry.get_client(LLMType.DEFAULT))}

    @property
    def tools(self) -> Dict[str, BaseTool]:
        return self._tools

    def _format_input(self, instruction: str, *args, **kwargs) -> WebSearchInput:
        query = instruction.strip()
        num_results = kwargs.get('num_results', 5)
        return WebSearchInput(query=query, num_results=num_results)

    def _process_output(self, output: WebSearchOutput) -> str:
        results = output.results
        formatted_results = "\n".join([
            f"Result {idx + 1}:\n"
            f"Title: {result.title}\n"
            f"URL: {result.url}\n"
            f"Description: {result.description}\n"
            f"Relevance Score: {result.relevance_score}"
            for idx, result in enumerate(results)
        ])
        return formatted_results

    def _run(self, llm_client: BaseLLMClient, input_data: WebSearchInput) -> WebSearchOutput:
        return self.tools["web_search"]._run(input_data)

    def _arun(self, llm_client: BaseLLMClient, input_data: WebSearchInput) -> WebSearchOutput:
        raise NotImplementedError("WebSearchAgent does not support async execution.")

    def setup(self):
        pass

    def teardown(self):
        pass
