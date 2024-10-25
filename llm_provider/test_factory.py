import pprint
import unittest
from unittest.mock import patch, Mock
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from config.openai_config import OpenAIConfig
from config.bedrock_config import BedrockConfig
from llm_provider.factory import LLMType, LLMFactory
from langchain.schema import HumanMessage, AIMessage

class SimpleOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        return text

class TestLLMFactory(unittest.TestCase):
    def setUp(self):
        self.configs = {
            LLMType.DEFAULT: [
                BedrockConfig(name="default_config", model="text-davinci-003", max_tokens=100),
                BedrockConfig(name="another_default", model="text-curie-001", max_tokens=50),
                OpenAIConfig(name="openai_default", model_name="text-davinci-003", max_tokens=100),
            ],
            LLMType.STRONG: [
                BedrockConfig(name="strong_config", model="text-davinci-002", temperature=0.2),
                OpenAIConfig(name="openai_strong", model_name="text-davinci-002", temperature=0.2),
            ],
        }
        self.factory = LLMFactory(self.configs)

    def test_add_config(self):
        new_config = BedrockConfig(name="new_config", model_name="text-ada-001")
        self.factory.add_config(LLMType.WEAK, new_config)
        self.assertIn(new_config, self.factory.configs[LLMType.WEAK])

    def test_remove_config(self):
        self.factory.remove_config(LLMType.DEFAULT, "another_default")
        self.assertEqual(len(self.factory.configs[LLMType.DEFAULT]), 2)
        self.assertEqual(self.factory.configs[LLMType.DEFAULT][0].name, "default_config")

    @patch("llm_provider.factory.ChatOpenAI")
    def test_create_openai_provider(self, mock_chatgpt):
        mock_chatgpt.return_value = Mock()
        config = self.configs[LLMType.DEFAULT][2]
        expected_config = config.dict(exclude_none=True)
        expected_config['llm_config'] = config.llm_config.model_dump(exclude_defaults=True)
        chain = self.factory.create_provider(LLMType.DEFAULT, name="openai_default")
        mock_chatgpt.assert_called_once_with(**expected_config["llm_config"])
        self.assertIsInstance(chain, mock_chatgpt.return_value.__class__)

    @patch("llm_provider.factory.ChatOpenAI")
    def test_create_openai_provider_with_tools(self, mock_chatgpt):
        mock_tool = Mock(spec=BaseTool)
        mock_chatgpt.return_value = Mock()
        chain = self.factory.create_provider(LLMType.STRONG, name="openai_strong", tools=[mock_tool])
        mock_chatgpt.return_value.bind_tools.assert_called_once_with(tools=[mock_tool])

    @patch("llm_provider.factory.ChatBedrock")
    def test_create_bedrock_provider(self, mock_chatgpt):
        mock_chatgpt.return_value = Mock()
        config = self.configs[LLMType.DEFAULT][0]
        expected_config = config.model_dump(exclude_none=True)
        expected_config['llm_config'] = config.llm_config.model_dump(exclude_defaults=True)
        chain = self.factory.create_provider(LLMType.DEFAULT, name="default_config")
        mock_chatgpt.assert_called_once_with(**expected_config['llm_config'])
        self.assertIsInstance(chain, mock_chatgpt.return_value.__class__)

    @patch("llm_provider.factory.ChatBedrock")
    def test_create_bedrock_provider_with_tools(self, mock_chatgpt):
        mock_tool = Mock(spec=BaseTool)
        mock_chatgpt.return_value = Mock()
        chain = self.factory.create_provider(LLMType.STRONG, tools=[mock_tool])
        mock_chatgpt.return_value.bind_tools.assert_called_once_with(tools=[mock_tool])

    def test_create_provider_with_output_parser(self):
        mock_parser = SimpleOutputParser()
        chain = self.factory.create_provider(LLMType.STRONG, output_parser=mock_parser)
        self.assertIn("ChatBedrock", str(chain))
        self.assertIn("SimpleOutputParser()", str(chain))
    
    def test_create_provider_with_prompt_template(self):
        mock_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="foo"),
            AIMessage(content="bar")
        ])
        chain = self.factory.create_provider(LLMType.STRONG, prompt_template=mock_prompt)
        self.assertIn("ChatPromptTemplate", str(chain))
        self.assertIn("ChatBedrock", str(chain))

    def test_create_provider_with_invalid_name(self):
        with self.assertRaises(ValueError):
            self.factory.create_provider(LLMType.DEFAULT, name="invalid_name")

    def test_create_provider_with_invalid_type(self):
        with self.assertRaises(ValueError):
            self.factory.create_provider(LLMType.VISION)

if __name__ == "__main__":
    unittest.main()
