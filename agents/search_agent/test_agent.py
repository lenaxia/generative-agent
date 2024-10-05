import unittest
from unittest.mock import Mock, patch
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from agents.search_agent.agent import SearchAgent
from llm_provider.factory import LLMFactory, LLMType
from config.openai_config import OpenAIConfig
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

class TestSearchAgent(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.message_bus = Mock()
        self.llm_factory = LLMFactory({})
        self.agent_id = "search_agent"

        # Configure the LLM provider
        openai_config = OpenAIConfig(
            name="openai",
            endpoint="http://192.168.5.74:8080/v1/",
            model_name="qwen2.5-32b-instruct",
        )
        self.llm_factory.add_config(LLMType.DEFAULT, openai_config)

        self.agent = SearchAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id)

    def test_tools(self):
        tools = self.agent.tools
        self.assertEqual(len(tools), 1)
        self.assertIsInstance(list(tools.values())[0], TavilySearchResults)

    @patch("langchain_openai.ChatOpenAI")
    def test_run_without_search(self, mock_chat_openai):
        mock_response = Mock()
        mock_response.tool_calls = []
        mock_response.content = "Hello, the weather is sunny today."
        mock_chat_openai.return_value.invoke.return_value = mock_response

        instruction = "What is the weather like today?"
        result = self.agent.run(instruction, llm_type=LLMType.DEFAULT)

        self.assertEqual(result, "Hello, the weather is sunny today.")
        mock_chat_openai.return_value.invoke.assert_called_once()

    @patch("langchain_openai.ChatOpenAI")
    def test_run_with_search(self, mock_chat_openai):
        mock_search_tool = Mock()
        mock_search_tool.run.return_value = "[{'url': 'https://example.com', 'content': 'Weather data'}]"
        self.agent.search_tool = mock_search_tool

        mock_response_1 = AIMessage(content=[{'id': 'toolu_01Y5EK4bw2LqsQXeaUv8iueF', 'input': {'query': 'weather in san francisco'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}])
        mock_response_2 = Mock()
        mock_response_2.content = "Based on the search results, the weather in San Francisco is ..."

        mock_chat_openai.return_value.invoke.side_effect = [mock_response_1, mock_response_2]

        instruction = "What is the weather like in San Francisco?"
        result = self.agent.run(instruction, llm_type=LLMType.DEFAULT)

        self.assertEqual(result, "Based on the search results, the weather in San Francisco is ...")
        mock_chat_openai.return_value.invoke.assert_called()
        mock_search_tool.run.assert_called_once_with("weather in san francisco")

if __name__ == "__main__":
    unittest.main()
