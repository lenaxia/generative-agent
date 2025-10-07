import unittest
from unittest.mock import Mock
from agents.search_agent.agent import SearchAgent
from llm_provider.factory import LLMFactory, LLMType
from common.message_bus import MessageBus
from logging import Logger
from agents.base_agent import AgentInput
from config.bedrock_config import BedrockConfig
from langchain_core.messages.ai import AIMessage

class TestSearchAgentIntegration(unittest.TestCase):
    def setUp(self):
        self.logger = Logger(__name__)

        # Configure the LLMFactory
        self.llm_factory = LLMFactory({})
        bedrock_config = BedrockConfig(
            name="bedrock", model='anthropic.claude-3-sonnet-20240229-v1:0'
        )
        self.llm_factory.add_config(LLMType.DEFAULT, bedrock_config)
        self.message_bus = Mock(MessageBus)
        self.agent_id = "search_agent"
        self.config = {"max_results": 2}
        self.agent = SearchAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id, self.config)

    def test_run_integration(self):
        instruction = "What is the capital of France?"
        history = ["This is some maybe relevant history", "This is some more history"]
        
        input = AgentInput(prompt=instruction, history=history)
        output = self.agent._run(input)
        
        print(output)

        self.assertIsInstance(output, dict)
        self.assertIn("agent", output)
        self.assertIn("messages", output["agent"])
        last_ai_message = output["agent"]["messages"][-1]
        self.assertIsInstance(last_ai_message, AIMessage)
        self.assertTrue("Paris" in last_ai_message.content)

if __name__ == "__main__":
    unittest.main()