import os
import unittest
from unittest.mock import patch, mock_open
from pathlib import Path
from supervisor.supervisor import Supervisor
from supervisor.supervisor_config import SupervisorConfig
from supervisor.request_manager import RequestManager, RequestMetadata
from supervisor.agent_manager import AgentManager
from supervisor.metrics_manager import MetricsManager
from common.message_bus import MessageBus
from llm_provider.factory import LLMFactory

current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, "test_config.yaml")

class TestSupervisor(unittest.TestCase):
    def setUp(self):
        self.config_file = config_file_path
        self.supervisor = Supervisor(self.config_file)

    @patch("supervisor.supervisor.initialize_config_manager")
    @patch("supervisor.supervisor.initialize_components")
    def test_init(self, mock_init_components, mock_init_config):
        Supervisor(self.config_file)
        mock_init_config.assert_called_once_with(self.config_file)
        mock_init_components.assert_called_once()

    @patch("supervisor.config_manager.ConfigManager")
    def test_initialize_config_manager(self, mock_config_manager):
        self.supervisor.initialize_config_manager(self.config_file)
        mock_config_manager.assert_called_once_with(self.config_file)

    @patch("supervisor.config_manager.ConfigManager.load_config")
    def test_initialize_config_manager_default_config(self, mock_load_config):
        default_config_file = self.config_file
        mock_open_file = mock_open(read_data="test data")
        with patch("builtins.open", mock_open_file):
            self.supervisor.initialize_config_manager()
            mock_load_config.assert_called_once()

#    @patch("supervisor.logging_config.configure_logging")
#    @patch("shared_tools.message_bus.MessageBus")
#    @patch("llm_provider.factory.LLMFactory")
#    @patch("supervisor.agent_manager.AgentManager")
#    @patch("supervisor.request_manager.RequestManager")
#    @patch("supervisor.metrics_manager.MetricsManager")
#    def test_initialize_components(self, mock_metrics_manager, mock_request_manager, mock_agent_manager, mock_llm_factory, mock_message_bus, mock_configure_logging):
#        mock_config = SupervisorConfig(log_level="INFO", log_file="test.log")
#        self.supervisor.config = mock_config
#        self.supervisor.initialize_components()
#        #mock_configure_logging.assert_called_once_with("INFO", "test.log")
#        mock_message_bus.assert_called_once()
#        mock_llm_factory.assert_called_once()
#        mock_agent_manager.assert_called_once_with(mock_config, mock_message_bus.return_value, mock_llm_factory.return_value)
#        mock_request_manager.assert_called_once_with(mock_agent_manager.return_value, mock_message_bus.return_value)
#        mock_metrics_manager.assert_called_once_with(mock_config)

    @patch("supervisor.agent_manager.AgentManager.register_agents")
    @patch("shared_tools.message_bus.MessageBus.start")
    def test_start(self, mock_message_bus_start, mock_register_agents):
        self.supervisor.agent_manager = AgentManager(SupervisorConfig(), MessageBus(), LLMFactory({}))
        self.supervisor.message_bus = MessageBus()
        self.supervisor.start()
        mock_register_agents.assert_called_once()
        mock_message_bus_start.assert_called_once()

    @patch("shared_tools.message_bus.MessageBus.stop")
    def test_stop(self, mock_message_bus_stop):
        self.supervisor.message_bus = MessageBus()
        self.supervisor.stop()
        mock_message_bus_stop.assert_called_once()

    @patch("supervisor.supervisor.start")
    @patch("supervisor.supervisor.stop")
    @patch("builtins.input", side_effect=["new", "Test instruction", "status", "stop"])
    def test_run(self, mock_input, mock_stop, mock_start):
        self.supervisor.request_manager = RequestManager(AgentManager(SupervisorConfig(), MessageBus(), LLMFactory({})), MessageBus())
        self.supervisor.run()
        mock_start.assert_called_once()
        mock_stop.assert_called_once()

    @patch("shared_tools.message_bus.MessageBus.is_running", return_value=True)
    @patch("supervisor.metrics_manager.MetricsManager.get_metrics", return_value={"test": "metrics"})
    def test_status(self, mock_get_metrics, mock_is_running):
        self.supervisor.message_bus = MessageBus()
        self.supervisor.metrics_manager = MetricsManager()
        status = self.supervisor.status()
        self.assertIsInstance(status, dict)
        self.assertTrue(status["running"])
        self.assertEqual(status["metrics"], {"test": "metrics"})

    def test_get_config_class(self):
        from config.openai_config import OpenAIConfig
        from config.anthropic_config import AnthropicConfig
        from config.bedrock_config import BedrockConfig

        self.assertEqual(self.supervisor.get_config_class("openai"), OpenAIConfig)
        self.assertEqual(self.supervisor.get_config_class("anthropic"), AnthropicConfig)
        self.assertEqual(self.supervisor.get_config_class("bedrock"), BedrockConfig)
        self.assertIsNone(self.supervisor.get_config_class("invalid"))

if __name__ == "__main__":
    unittest.main()
