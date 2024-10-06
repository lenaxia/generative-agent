import unittest
from unittest.mock import Mock, patch
from agents.weather_agent.agent import WeatherAgent
from llm_provider.factory import LLMFactory, LLMType
#from config.openai_config import OpenAIConfig
from config.bedrock_config import BedrockConfig
from shared_tools.message_bus import MessageBus

class WeatherAgentTest(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.llm_factory = LLMFactory({})
        self.message_bus = MessageBus()
        self.agent_id = "weather_agent"

        # Configure the LLM provider
        #openai_config = OpenAIConfig(
        #    name="openai",
        #    endpoint="http://192.168.5.74:8080/v1/",
        #    model_name="qwen2.5-32b-instruct",
        #    temperature=0.7,
        #    max_tokens=512,
        #    top_p=1.0,
        #    api_key="YOUR_API_KEY",
        #)
        bedrock_config = BedrockConfig(
                name="bedrock", model_id='anthropic.claude-3-sonnet-20240229-v1:0', model_kwargs={'temperature': 0}
            )
        self.llm_factory.add_config(LLMType.DEFAULT, bedrock_config)

        self.agent = WeatherAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id)

    @patch("langchain_aws.ChatBedrock")
    def test_agent_run(self, mock_chat):
        instruction = "What is the weather in Seattle?"
        expected_output = "It's always sunny in San Francisco"

        # Mock the ChatOpenAI instance
        mock_chat_instance = mock_chat.return_value
        mock_chat_instance.stream.return_value = [
            {"messages": [("user", instruction), ("assistant", expected_output)]}
        ]

        result = self.agent.run(instruction)

        print(str(result))

        # Assert that the output is as expected
        self.assertEqual(result, expected_output)

        # Assert that the ChatOpenAI instance was created with the correct parameters
        mock_chat.assert_called_with(
            model_id='anthropic.claude-3-sonnet-20240229-v1:0', model_kwargs={'temperature': 0}
            #base_url="http://192.168.5.74:8080/v1/",
            #model_name="gpt-4",
            #temperature=0.7,
            #max_tokens=512,
            #top_p=1.0,
            #api_key="YOUR_API_KEY",
        )

if __name__ == "__main__":
    unittest.main()
