from logging import Logger
from typing import Any, Dict, List
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import Runnable
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from llm_provider.factory import LLMFactory, LLMType
from shared_tools.message_bus import MessageBus
from agents.base_agent import BaseAgent
from langchain_community.tools.tavily_search import TavilySearchResults

class SearchAgent(BaseAgent):
    def __init__(self, logger: Logger, llm_factory: LLMFactory, message_bus: MessageBus, agent_id: str, config: Dict = None):
        super().__init__(logger, llm_factory, message_bus, agent_id, config)
        self.search_tool = TavilySearchResults(max_results=2)
        self.prompt_template = ChatPromptTemplate.from_template("{instruction}\n\nSearch Quality Reflection: {search_quality_reflection}\nSearch Quality Score: {search_quality_score:0.2f}\n\nSearch Results:\n{search_results}")
        self.output_parser = StrOutputParser()

    @property
    def tools(self) -> Dict[str, BaseTool]:
        return {"search": self.search_tool}

    def _run(self, llm_provider, instruction: str) -> Any:
        messages = [HumanMessage(content=instruction)]
        response = llm_provider.invoke(messages)

        if response.tool_calls:
            messages.append(response)  # Add the initial AI response with tool calls

            for tool_call in response.tool_calls:
                messages.append(str(tool_call))  # Add the initial AI response with tool calls
                if tool_call["name"] == "tavily_search_results_json":
                    search_query = tool_call["args"]["query"]
                    search_results = self.search_tool.run(search_query)
                    messages.append(str(ToolMessage(content=search_results, name="tavily_search_results_json")))

            final_response = llm_provider.invoke(messages)
            messages.append(final_response)

        return final_response

    def _arun(self, llm_provider, instruction: str) -> Any:
        raise NotImplementedError("Asynchronous execution not supported.")

    def _format_input(self, instruction: str, *args, **kwargs) -> str:
        return instruction

    def _process_output(self, output: str, *args, **kwargs) -> str:
        return output

    def _select_llm_provider(self, llm_type: LLMType, **kwargs) -> Runnable:
        return self.llm_factory.create_provider(llm_type, tools=self.tools.values(), **kwargs)

    def setup(self):
        pass

    def teardown(self):
        pass
