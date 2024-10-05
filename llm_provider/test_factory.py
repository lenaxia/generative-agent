import pprint
import unittest
from unittest.mock import patch, Mock
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from config.base_config import BaseConfig
from config.openai_config import OpenAIConfig
from typing import Dict, List, Union, Type
from pydantic import BaseModel
from llm_provider.factory import LLMType, LLMFactory
from langchain.schema import HumanMessage, AIMessage

class SimpleOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        return text

class TestLLMFactory(unittest.TestCase):
    def setUp(self):
        self.configs = {
            LLMType.DEFAULT: [
                OpenAIConfig(name="default_config", model_name="text-davinci-003", max_tokens=100),
                OpenAIConfig(name="another_default", model_name="text-curie-001", max_tokens=50),
            ],
            LLMType.STRONG: [
                OpenAIConfig(name="strong_config", model_name="text-davinci-002", temperature=0.2),
            ],
        }
        self.factory = LLMFactory(self.configs)

    def test_add_config(self):
        new_config = OpenAIConfig(name="new_config", model_name="text-ada-001")
        self.factory.add_config(LLMType.WEAK, new_config)
        self.assertIn(new_config, self.factory.configs[LLMType.WEAK])

    def test_remove_config(self):
        self.factory.remove_config(LLMType.DEFAULT, "another_default")
        self.assertEqual(len(self.factory.configs[LLMType.DEFAULT]), 1)
        self.assertEqual(self.factory.configs[LLMType.DEFAULT][0].name, "default_config")

    @patch("llm_provider.factory.ChatOpenAI")
    def test_create_provider(self, mock_chatgpt):
        mock_chatgpt.return_value = Mock()
        chain = self.factory.create_provider(LLMType.DEFAULT, name="default_config")
        mock_chatgpt.assert_called_once_with(
            model_name="text-davinci-003",
            temperature=0.7,
            max_tokens=100,
            top_p=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=1,
            logprobs=False,
            top_logprobs=None,
            model_kwargs={"stream": False, "request_kwargs": None},
            api_key="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",
        )
        self.assertIsInstance(chain, mock_chatgpt.return_value.__class__)

    @patch("llm_provider.factory.ChatOpenAI")
    def test_create_provider_with_tools(self, mock_chatgpt):
        mock_tool = Mock(spec=BaseTool)
        mock_chatgpt.return_value = Mock()
        chain = self.factory.create_provider(LLMType.STRONG, tools=[mock_tool])
        mock_chatgpt.return_value.bind_tools.assert_called_once_with(tools=[mock_tool])

    def test_create_provider_with_output_parser(self):
        mock_parser = SimpleOutputParser()
        chain = self.factory.create_provider(LLMType.STRONG, output_parser=mock_parser)
        self.assertIn("ChatOpenAI", str(chain))
        self.assertIn("SimpleOutputParser()", str(chain))
    
    def test_create_provider_with_prompt_template(self):
        mock_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="foo"),
            AIMessage(content="bar")
        ])
        chain = self.factory.create_provider(LLMType.STRONG, prompt_template=mock_prompt)
        self.assertIn("ChatPromptTemplate", str(chain))
        self.assertIn("ChatOpenAI", str(chain))

    def test_create_provider_with_invalid_name(self):
        with self.assertRaises(ValueError):
            self.factory.create_provider(LLMType.DEFAULT, name="invalid_name")

    def test_create_provider_with_invalid_type(self):
        with self.assertRaises(ValueError):
            self.factory.create_provider(LLMType.VISION)

if __name__ == "__main__":
    unittest.main()
