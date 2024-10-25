import unittest
from unittest.mock import patch, Mock
from typing import Dict, Any

from supervisor.request_manager import RequestManager, RequestModel
from common.task_graph import TaskGraph
from supervisor.agent_manager import AgentManager
from supervisor.supervisor_config import SupervisorConfig
from shared_tools.message_bus import MessageBus
from llm_provider.factory import LLMFactory, LLMType
from config.bedrock_config import BedrockConfig
from agents.hello_world_agent.agent import HelloWorldAgent
from agents.weather_agent.agent import WeatherAgent
from agents.planning_agent.agent import PlanningAgent

class RequestManagerIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.llmconfigs = {
            LLMType.DEFAULT: [
                BedrockConfig(name="default_config", model_id='anthropic.claude-3-sonnet-20240229-v1:0', model_kwargs={'temperature': 0}),
            ],
        }
        self.logger = Mock()
        self.config = SupervisorConfig()
        self.message_bus = MessageBus()
        self.llm_factory = LLMFactory(self.llmconfigs)
        self.agent_manager = AgentManager(self.config, self.message_bus, self.llm_factory)
        self.agent_manager.register_agent(HelloWorldAgent(self.logger, self.llm_factory, self.message_bus, agent_id="HelloWorldAgent"))
        self.agent_manager.register_agent(WeatherAgent(self.logger, self.llm_factory, self.message_bus, agent_id="WeatherAgent"))
        self.agent_manager.register_agent(PlanningAgent(self.logger, self.llm_factory, self.message_bus, agent_id="PlanningAgent"))
        self.request_manager = RequestManager(self.agent_manager, self.message_bus)

    @patch("supervisor.request_manager.RequestManager.persist_request")
    def test_handle_request(self, mock_persist_request):
        # Arrange
        request = RequestModel(instructions="What is the weather in Seattle?")

        # Act
        request_id = self.request_manager.handle_request(request)

        # Assert
        self.assertIsInstance(request_id, str)
        self.assertIn(request_id, self.request_manager.request_map)
        self.assertIsInstance(self.request_manager.request_map[request_id], TaskGraph)
        #mock_persist_request.assert_called_once_with(request_id)

    @patch("supervisor.request_manager.MessageBus.publish")
    def test_monitor_progress(self, mock_publish):
        # Arrange
        request = RequestModel(instructions="What is the weather in Seattle?")
        request_id = self.request_manager.handle_request(request)

        # Act
        self.request_manager.monitor_progress(request_id)

        # Assert
        mock_publish.assert_called()

    @patch("supervisor.request_manager.MessageBus.publish")
    def test_handle_task_response(self, mock_publish):
        # Arrange
        request = RequestModel(instructions="What is the weather in Seattle?")
        request_id = self.request_manager.handle_request(request)
        task_graph = self.request_manager.request_map[request_id]
        task_node = list(task_graph.nodes.values())[0]
        response = {
            "request_id": request_id,
            "task_id": task_node.task_id,
            "result": "The weather in Seattle is currently sunny with a high of 75°F.",
            "status": "COMPLETED"
        }

        # Act
        self.request_manager.handle_task_response(response)

        # Assert
        self.assertEqual(task_node.status, "COMPLETED")
        self.assertEqual(task_node.result, response["result"])
        mock_publish.assert_called()

if __name__ == "__main__":
    unittest.main()
