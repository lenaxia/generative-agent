import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import yaml
from supervisor.agent_manager import AgentManager
from supervisor.supervisor_config import SupervisorConfig
from common.message_bus import MessageBus
from agents.base_agent import BaseAgent
from agents.hello_world_agent.agent import HelloWorldAgent
from agents.weather_agent.agent import WeatherAgent
from llm_provider.factory import LLMFactory, LLMType
from config.openai_config import OpenAIConfig

class TestAgentManager(unittest.TestCase):
    def setUp(self):
        self.supervisor_config = SupervisorConfig()
        self.message_bus = MessageBus()
        self.llm_factory = LLMFactory({})
        self.openai_config = OpenAIConfig(
            name="openai",
            endpoint="http://example.com",
            model_name="text-davinci-003",
            max_tokens=100,
        )
        self.llm_factory.add_config(LLMType.DEFAULT, self.openai_config)
        self.agent_manager = AgentManager(self.supervisor_config, self.message_bus, self.llm_factory)

    def test_register_agent(self):
        mock_agent = Mock(spec=BaseAgent, agent_id="test_agent")
        self.agent_manager.register_agent(mock_agent)
        self.assertIn("test_agent", self.agent_manager.agent_registry)

    def test_unregister_agent(self):
        mock_agent = Mock(spec=BaseAgent, agent_id="test_agent")
        self.agent_manager.register_agent(mock_agent)
        self.agent_manager.unregister_agent("test_agent")
        self.assertNotIn("test_agent", self.agent_manager.agent_registry)

    def test_get_agent(self):
        mock_agent = Mock(spec=BaseAgent, agent_id="test_agent")
        self.agent_manager.register_agent(mock_agent)
        retrieved_agent = self.agent_manager.get_agent("test_agent")
        self.assertEqual(retrieved_agent, mock_agent)

    def test_get_agents(self):
        mock_agent1 = Mock(spec=BaseAgent, agent_id="test_agent1")
        mock_agent2 = Mock(spec=BaseAgent, agent_id="test_agent2")
        self.agent_manager.register_agent(mock_agent1)
        self.agent_manager.register_agent(mock_agent2)
        agents = self.agent_manager.get_agents()
        self.assertCountEqual(agents, [mock_agent1, mock_agent2])

    @patch('supervisor.agent_manager.Path.exists', return_value=True)
    @patch('supervisor.agent_manager.yaml.safe_load', return_value={'config': {'test_key': 'test_value'}})
    def test_register_agents(self, mock_safe_load, mock_exists):
        self.agent_manager.register_agents()
        self.assertIn('HelloWorldAgent', self.agent_manager.agent_registry)
        self.assertIn('WeatherAgent', self.agent_manager.agent_registry)
        hello_world_agent = self.agent_manager.get_agent('HelloWorldAgent')
        self.assertIsInstance(hello_world_agent, HelloWorldAgent)
        weather_agent = self.agent_manager.get_agent('WeatherAgent')
        self.assertIsInstance(weather_agent, WeatherAgent)

if __name__ == '__main__':
    unittest.main()
