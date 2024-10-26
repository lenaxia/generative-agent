import unittest
from typing import Dict
from unittest.mock import Mock, patch
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import Runnable
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from llm_provider.factory import LLMFactory, LLMType
from common.message_bus import MessageBus
from agents.base_agent import BaseAgent, AgentInput
from langchain_community.tools.tavily_search import TavilySearchResults
from agents.search_agent.agent import SearchAgent

class TestSearchAgent(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.llm_factory = Mock(spec=LLMFactory)
        self.message_bus = Mock(spec=MessageBus)
        self.agent_id = "search_agent"
        self.config = {"max_results": 2}

    def test_init(self):
        agent = SearchAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id, self.config)
        self.assertIsInstance(agent.search_tool, TavilySearchResults)
        #self.assertIsInstance(agent.prompt_template, ChatPromptTemplate)
        #self.assertIsInstance(agent.output_parser, StrOutputParser)

    def test_format_input(self):
        agent = SearchAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id, self.config)
        prompt = "What is the capital of France?"
        history = ["This is some maybe relevant history", "This is some more history"]
        formatted_input = agent._format_input({"prompt": prompt, "history": history})
        expected_input = AgentInput(prompt=prompt, history=history)
        self.assertEqual(formatted_input, expected_input)

    def test_process_output(self):
        agent = SearchAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id, self.config)
        output = {
            "agent": {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "The capital of France is Paris."
                    }
                ]
            }
        }
        input: Dict = dict(
            task_id="task_1",
            task_name="fetch_data",
            request_id="request_1",
            agent_id="agent_1",
            task_type="fetch_data",
            prompt="Fetch data from API",
            status="pending",
            inbound_edges=[],
            outbound_edges=[],
            include_full_history=False,
        )
        processed_output = agent._process_output(input, output)
        expected_output = "The capital of France is Paris."
        self.assertEqual(processed_output, expected_output)

    def test_setup_and_teardown(self):
        agent = SearchAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id, self.config)
        agent.setup()  # No exceptions should be raised
        agent.teardown()  # No exceptions should be raised

if __name__ == '__main__':
    unittest.main()