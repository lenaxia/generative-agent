import unittest
from unittest.mock import Mock, patch
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import Runnable
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from llm_provider.factory import LLMFactory, LLMType
from shared_tools.message_bus import MessageBus
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
        self.assertIsInstance(agent.prompt_template, ChatPromptTemplate)
        self.assertIsInstance(agent.output_parser, StrOutputParser)

    @patch.object(TavilySearchResults, 'run')
    def test_run(self, mock_search_tool):
        instruction = "What is the capital of France?"
        search_results = "Paris is the capital of France."
        mock_search_tool.return_value = search_results

        llm_provider = Mock(spec=Runnable)
        initial_response = AIMessage(content="I will search for the capital of France.", tool_calls=[{"name": "tavily_search_results_json", "id": "123","args": {"query": "capital of france"}}])
        final_response = AIMessage(content="The capital of France is Paris.")
        llm_provider.invoke.side_effect = [initial_response, final_response]
        
        input = AgentInput(prompt=instruction)

        agent = SearchAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id, self.config)
        output = agent._run(input)

        self.assertEqual(output, "The capital of France is Paris.")
        mock_search_tool.assert_called_once_with("capital of france")
        #llm_provider.invoke.assert_has_calls([
        #    unittest.mock.call([HumanMessage(content=instruction)]),
        #    unittest.mock.call([
        #        HumanMessage(content=instruction),
        #        initial_response,
        #        ToolMessage(content=search_results, name="tavily_search_results_json", tool_call_id="123")
        #    ])
        #])

    def test_format_input(self):
        agent = SearchAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id, self.config)
        instruction = "What is the capital of France?"
        formatted_input = agent._format_input(instruction)
        self.assertEqual(formatted_input, instruction)

    def test_process_output(self):
        agent = SearchAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id, self.config)
        output = "The capital of France is Paris."
        processed_output = agent._process_output(output)
        self.assertEqual(processed_output, output)

    def test_setup_and_teardown(self):
        agent = SearchAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id, self.config)
        agent.setup()  # No exceptions should be raised
        agent.teardown()  # No exceptions should be raised

if __name__ == '__main__':
    unittest.main()