import unittest
from unittest.mock import Mock
from supervisor.agent_manager import AgentManager
from supervisor.supervisor_config import SupervisorConfig
from llm_provider.factory import LLMFactory, LLMType
from config.openai_config import OpenAIConfig
from common.message_bus import MessageBus
from agents.base_agent import AgentInput

class WeatherAgentManagerIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.supervisor_config = SupervisorConfig()
        self.message_bus = MessageBus()
        self.llm_factory = LLMFactory({})

        # Configure the LLM provider
        openai_config = OpenAIConfig(
            name="openai",
            endpoint="http://192.168.5.74:8080/v1/",
            model_name="qwen2.5-32b-instruct",
        )
        self.llm_factory.add_config(LLMType.DEFAULT, openai_config)

        # Create the AgentManager
        self.agent_manager = AgentManager(self.supervisor_config, self.message_bus, self.llm_factory)
        self.agent_manager.register_agents()

    def test_weather_agent(self):
        weather_agent = self.agent_manager.get_agent("WeatherAgent")
        self.assertIsNotNone(weather_agent)

        # Create prompt template
        prompt_template = weather_agent.prompt_template

        # Create LLM provider
        llm_provider = self.llm_factory.create_provider(
            llm_type=LLMType.DEFAULT,
            name="openai",
            prompt_template=prompt_template,
        )

        # Check weather for several cities
        cities = ["Seattle", "New York", "London", "Tokyo"]
        for city in cities:
            instruction = f"What is the current weather in {city}?"
            input = AgentInput(prompt=instruction)
            result = weather_agent._run(input)

            print(f"Weather for {city}: {result}")

            self.assertIsNotNone(result)
            self.assertIn("The weather is currently", result)

if __name__ == "__main__":
    unittest.main()
