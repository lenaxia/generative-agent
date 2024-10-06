import unittest
from unittest.mock import Mock, patch
from supervisor.request_manager import RequestManager, RequestModel
from supervisor.task_graph import TaskGraph, TaskNode
from supervisor.task_models import Task, TaskStatus
from shared_tools.message_bus import MessageBus, MessageType
from supervisor.agent_manager import AgentManager
from supervisor.supervisor_config import SupervisorConfig
from llm_provider.factory import LLMFactory, LLMType
from agents.hello_world_agent.agent import HelloWorldAgent
from agents.weather_agent.agent import WeatherAgent
from config.bedrock_config import BedrockConfig 

class TestRequestManager(unittest.TestCase):
    def setUp(self):
        self.llmconfigs = {
            LLMType.DEFAULT: [
                BedrockConfig(name="default_config", model_name="text-davinci-003", max_tokens=100),
                BedrockConfig(name="another_default", model_name="text-curie-001", max_tokens=50),
            ],
            LLMType.STRONG: [
                BedrockConfig(name="strong_config", model_name="text-davinci-002", temperature=0.2),
            ],
        }
        self.config = SupervisorConfig()
        self.message_bus = MessageBus()
        self.llm_factory = LLMFactory(self.llmconfigs)
        self.agent_manager = AgentManager(self.config, self.message_bus, self.llm_factory)
        self.agent_manager.register_agent(HelloWorldAgent(None, self.llm_factory, self.message_bus, agent_id="HelloWorldAgent"))
        self.agent_manager.register_agent(WeatherAgent(None, self.llm_factory, self.message_bus, agent_id="WeatherAgent"))
        self.request_manager = RequestManager(self.agent_manager, self.message_bus)

    def test_handle_request(self):
        # Arrange
        request = RequestModel(instructions="Test instruction")

        # Act
        request_id = self.request_manager.handle_request(request)

        # Assert
        self.assertIsInstance(request_id, str)
        self.assertIn(request_id, self.request_manager.request_map)
        self.assertIsInstance(self.request_manager.request_map[request_id], TaskGraph)

    def test_create_task_graph(self):
        # Arrange
        agents = self.agent_manager.get_agents()

        # Act
        task_graph = self.request_manager.create_task_graph("Test instruction", agents)

        # Assert
        self.assertIsInstance(task_graph, TaskGraph)

    def test_monitor_progress(self):
        # Arrange
        request = RequestModel(instructions="Test instruction")
        request_id = self.request_manager.handle_request(request)
        task_graph = self.request_manager.request_map[request_id]

        # Act
        self.request_manager.monitor_progress(request_id)

        # Assert
        # Add your assertions here to check the expected behavior after monitoring progress

    def test_handle_task_response(self):
        # Arrange
        request = RequestModel(instructions="Test instruction")
        request_id = self.request_manager.handle_request(request)
        task_graph = self.request_manager.request_map[request_id]
        task_node = list(task_graph.nodes.values())[0]
        response = {
            "request_id": request_id,
            "task_id": task_node.task.id,
            "result": "Test result"
        }

        # Act
        self.request_manager.handle_task_response(response)

        # Assert
        self.assertEqual(task_node.task.status, TaskStatus.COMPLETED)
        self.assertEqual(task_node.task.result, "Test result")

    def test_handle_agent_error(self):
        # Arrange
        request = RequestModel(instructions="Test instruction")
        request_id = self.request_manager.handle_request(request)
        task_graph = self.request_manager.request_map[request_id]
        task_node = list(task_graph.nodes.values())[0]
        error = {
            "request_id": request_id,
            "task_id": task_node.task.id,
            "error_message": "Test error"
        }

        # Act
        self.request_manager.handle_agent_error(error)

        # Assert
        self.assertEqual(task_node.task.status, TaskStatus.FAILED)
        self.assertEqual(task_node.task.error, "Test error")

    @patch("supervisor.request_manager.time.sleep")
    def test_retry_failed_task(self, mock_sleep):
        # Arrange
        request = RequestModel(instructions="Test instruction")
        request_id = self.request_manager.handle_request(request)
        task_graph = self.request_manager.request_map[request_id]
        task_node = list(task_graph.nodes.values())[0]
        task_node.task.status = TaskStatus.FAILED
        self.request_manager.config.max_retries = 3
        self.request_manager.config.retry_delay = 0.1
        self.request_manager.config.metrics_manager.update_metrics(request_id, {"retries": 1})

        # Act
        self.request_manager.retry_failed_task(task_node.task.id, request_id)

        # Assert
        self.request_manager.config.metrics_manager.update_metrics.assert_called_with(request_id, {"retries": 2})
        mock_sleep.assert_called_once_with(0.1)
        self.message_bus.publish.assert_called_once()

    def test_get_request_status(self):
        # Arrange
        request = RequestModel(instructions="Test instruction")
        request_id = self.request_manager.handle_request(request)
        task_graph = self.request_manager.request_map[request_id]

        # Act
        status = self.request_manager.get_request_status(request_id)

        # Assert
        expected_status = {task.id: TaskStatus.PENDING for task in task_graph.tasks}
        self.assertEqual(status, expected_status)

    def test_delegate_task(self):
        # Arrange
        request = RequestModel(instructions="Test instruction")
        request_id = self.request_manager.handle_request(request)
        task_graph = self.request_manager.request_map[request_id]
        task_node = list(task_graph.nodes.values())[0]

        # Act
        self.request_manager.delegate_task(task_node.task, request_id)

        # Assert
        self.message_bus.publish.assert_called_once()

    def test_handle_request_completion(self):
        # Arrange
        request = RequestModel(instructions="Test instruction")
        request_id = self.request_manager.handle_request(request)
        task_graph = self.request_manager.request_map[request_id]

        # Simulate task completion
        for node in task_graph.nodes.values():
            node.task.status = TaskStatus.COMPLETED

        # Act
        self.request_manager.handle_request_completion(request_id)

        # Assert
        self.request_manager.config.metrics_manager.update_metrics.assert_called_once()
        self.request_manager.persist_request.assert_called_once_with(request_id)

    def test_handle_request_failure(self):
        # Arrange
        request = RequestModel(instructions="Test instruction")
        request_id = self.request_manager.handle_request(request)
        error_message = "Test error"

        # Act
        self.request_manager.handle_request_failure(request_id, error_message)

        # Assert
        self.request_manager.config.metrics_manager.update_metrics.assert_called_once()
        self.request_manager.persist_request.assert_called_once_with(request_id)

    def test_persist_request(self):
        # Arrange
        request = RequestModel(instructions="Test instruction")
        request_id = self.request_manager.handle_request(request)
        task_graph = self.request_manager.request_map[request_id]
        metrics_mock = {"key": "value"}
        self.request_manager.config.metrics_manager.get_metrics.return_value = metrics_mock

        # Act
        self.request_manager.persist_request(request_id)

        # Assert
        self.request_manager.config.metrics_manager.persist_metrics.assert_called_once()

    def test_load_request(self):
        # Arrange
        request_id = "test_request"
        request_data = {"key": "value"}
        self.request_manager.config.metrics_manager.load_metrics.return_value = request_data
    
        # Act
        loaded_data = self.request_manager.load_request(request_id)
    
        # Assert
        self.assertEqual(loaded_data, request_data)
        self.request_manager.config.metrics_manager.load_metrics.assert_called_once_with(request_id)
    
    def test_check_conditions(self):
        # Arrange
        request = RequestModel(instructions="Test instruction")
        request_id = self.request_manager.handle_request(request)
        task_graph = self.request_manager.request_map[request_id]
        node = list(task_graph.nodes.values())[0]
    
        # Act
        result = self.request_manager.check_conditions(node, task_graph)
    
        # Assert
        self.assertTrue(result)  # Assuming the default implementation returns True
    
    def test_handle_request_with_exception(self):
        # Arrange
        request = RequestModel(instructions="Test instruction")
        self.request_manager.create_task_graph = Mock(side_effect=Exception("Test exception"))
    
        # Act & Assert
        with self.assertRaises(Exception):
            self.request_manager.handle_request(request)
    
    def test_monitor_progress_with_invalid_request_id(self):
        # Arrange
        invalid_request_id = "invalid_id"
    
        # Act & Assert
        with self.assertLogs(level="ERROR") as cm:
            self.request_manager.monitor_progress(invalid_request_id)
        self.assertIn(f"Request '{invalid_request_id}' not found.", cm.output)
    
    def test_handle_task_response_with_invalid_request_id(self):
        # Arrange
        invalid_request_id = "invalid_id"
        response = {
            "request_id": invalid_request_id,
            "task_id": "task1",
            "result": "Test result"
        }
    
        # Act & Assert
        with self.assertLogs(level="ERROR") as cm:
            self.request_manager.handle_task_response(response)
        self.assertIn(f"Request '{invalid_request_id}' not found.", cm.output)
    
    def test_handle_agent_error_with_invalid_request_id(self):
        # Arrange
        invalid_request_id = "invalid_id"
        error = {
            "request_id": invalid_request_id,
            "task_id": "task1",
            "error_message": "Test error"
        }
    
        # Act & Assert
        with self.assertLogs(level="ERROR") as cm:
            self.request_manager.handle_agent_error(error)
        self.assertIn(f"Request '{invalid_request_id}' not found.", cm.output)
    
if __name__ == '__main__':
    unittest.main()
