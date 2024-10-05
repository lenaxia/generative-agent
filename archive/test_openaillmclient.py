import unittest
from unittest.mock import patch, Mock
from config.openai_config import OpenAIConfig
from llm_provider.base_client import OpenAILLMClient
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from typing import Any

class TestTool(BaseTool):
    name: str = "test_tool"  # Add type annotation
    description: str = "A simple test tool"  # Add type annotation

    def _run(self, x: Any) -> Any:
        return x

    def _arun(self, x: Any) -> Any:
        raise NotImplementedError("This tool does not support async execution")

class TestOpenAILLMClient(unittest.TestCase):
    def setUp(self):
        self.openai_config = OpenAIConfig(
            name="test_openai_llm_client",
            model_name="qwen2.5-32b-instruct",
            endpoint="http://192.168.5.74:8080/v1/chat/completions",
            temperature=0.1,
            api_key="random_string"
        )
        self.openai_llm_client = OpenAILLMClient(config=self.openai_config, name="openai_llm_client")
        self.prompt_template = ChatPromptTemplate.from_template("{input}")
        self.tool = TestTool()
        self.openai_llm_client.initialize_responder(self.prompt_template, [self.tool])

    def test_openai_llm_client_initialization(self):
        self.assertIsInstance(self.openai_llm_client.config, OpenAIConfig)
        self.assertEqual(self.openai_llm_client.name, "openai_llm_client")
        self.assertIsInstance(self.openai_llm_client.model, ChatOpenAI)
        self.assertEqual(self.openai_llm_client.model.model_name, self.openai_config.model_name)
        self.assertEqual(self.openai_llm_client.model.endpoint, self.openai_config.endpoint)
        self.assertEqual(self.openai_llm_client.model.temperature, self.openai_config.temperature)
        self.assertEqual(self.openai_llm_client.model.api_key, self.openai_config.api_key)

    @patch('langchain.chat_models.ChatOpenAI')
    def test_openai_llm_client_respond(self, mock_chat_openai):
        mock_chat_openai_instance = Mock()
        mock_chat_openai.return_value = mock_chat_openai_instance

        state = [{"role": "user", "content": "What is the meaning of life?"}]
        expected_response = {"result": "The meaning of life is to find happiness and fulfillment."}

        mock_chat_openai_instance.generate.return_value = expected_response

        response = self.openai_llm_client.respond(state)

        self.assertEqual(response, expected_response)
        mock_chat_openai.assert_called_once_with(
            model_name="qwen2.5-32b-instruct",
            endpoint="http://192.168.5.74:8080/v1/chat/completions",
            temperature=0.1,
            api_key="random_string",
        )
        mock_chat_openai_instance.generate.assert_called_once_with(messages=state, callbacks=mock.ANY)

    def test_openai_llm_client_integration(self):
        state = [{"role": "user", "content": "What is the meaning of life?"}]

        response = self.openai_llm_client.respond(state)

        self.assertIsInstance(response, dict)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], str)
        self.assertGreater(len(response["result"]), 0)

if __name__ == '__main__':
    unittest.main()
