import unittest
from unittest.mock import Mock
from langchain.prompts import ChatPromptTemplate
from agents.hello_world_agent.agent import HelloWorldAgent
from llm_provider.factory import LLMFactory, LLMType
from config.openai_config import OpenAIConfig
from shared_tools.message_bus import MessageBus

class HelloWorldAgentIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.llm_factory = LLMFactory({})
        self.message_bus = MessageBus()
        self.agent_id = "hello_world_agent"
        self.agent = HelloWorldAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id)

        # Configure the LLM provider
        openai_config = OpenAIConfig(
            name="openai",
            endpoint="http://192.168.5.74:8080/v1/",
            model_name="qwen2.5-32b-instruct",
        )
        self.llm_factory.add_config(LLMType.DEFAULT, openai_config)

    def test_agent_invoke(self):
        instruction = "Please respond only with 'Hello, World!' and nothing else"
        expected_output = "Hello, World!"

        # Create prompt template
        prompt_template = ChatPromptTemplate.from_template("{instruction}")

        # Create LLM provider
        llm_provider = self.llm_factory.create_provider(
            llm_type=LLMType.DEFAULT,
            name="openai",
            prompt_template=prompt_template,
        )

        # Run the agent
        result = self.agent._run(llm_provider, instruction)

        # Assert that the output is as expected
        self.assertEqual(result.content, expected_output)

if __name__ == "__main__":
    unittest.main()
