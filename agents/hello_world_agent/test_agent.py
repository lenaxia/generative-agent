import unittest
from unittest.mock import Mock
from agents.hello_world_agent.agent import HelloWorldAgent
from llm_provider.factory import LLMFactory, LLMType
from shared_tools.message_bus import MessageBus

class TestHelloWorldAgent(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.llm_factory = LLMFactory({})
        self.message_bus = MessageBus()
        self.agent_id = "hello_world_agent"
        self.agent = HelloWorldAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id)

    def test_run(self):
        instruction = "Hello, World!"
        expected_output = "Hello, World!"

        # Mock the LLM provider
        mock_provider = Mock()
        mock_provider.invoke.return_value = expected_output
        self.llm_factory.create_provider = Mock(return_value=mock_provider)

        result = self.agent.run(instruction, history=[])
        self.assertEqual(result, expected_output)

    def test_format_input(self):
        instruction = "Hello, Test!"
        formatted_input = self.agent._format_input(instruction)
        self.assertEqual(formatted_input, instruction)

    def test_process_output(self):
        output = "Hello, Output!"
        processed_output = self.agent._process_output(output)
        self.assertEqual(processed_output, output)

if __name__ == '__main__':
    unittest.main()
