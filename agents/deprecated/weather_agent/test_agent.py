import unittest
from unittest.mock import Mock
from agents.weather_agent.agent import WeatherAgent
from llm_provider.factory import LLMFactory, LLMType
#from config.openai_config import OpenAIConfig
from config.bedrock_config import BedrockConfig
from common.message_bus import MessageBus

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
                name="bedrock", model='anthropic.claude-3-sonnet-20240229-v1:0', model_kwargs={'temperature': 0}
            )
        self.llm_factory.add_config(LLMType.DEFAULT, bedrock_config)

        self.agent = WeatherAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id)


    def test_agent_run_city(self):
        prompt = "What is the weather in Seattle?"
        history = ["This is some maybe relevant history", "This is some more history"]
        formatted_input = self.agent._format_input({"prompt": prompt, "history": history})

        output = self.agent._run(formatted_input)

        result = output["agent"]["messages"][0].content
        print(str(result))

        # Assert that the output is as expected
        self.assertIn("Seattle",result)
        
    def test_agent_run_zipcode(self):
        prompt = "What is the weather in 98104?"
        history = ["This is some maybe relevant history", "This is some more history"]
        formatted_input = self.agent._format_input({"prompt": prompt, "history": history})

        output = self.agent._run(formatted_input)

        result = output["agent"]["messages"][0].content
        print(str(result))

        # Assert that the output is as expected
        self.assertIn("Seattle",result)

if __name__ == "__main__":
    unittest.main()
