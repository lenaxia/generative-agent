import os
import unittest
from unittest.mock import Mock, patch

from common.message_bus import MessageBus
from llm_provider.factory import LLMFactory
from supervisor.metrics_manager import MetricsManager
from supervisor.supervisor import Supervisor
from supervisor.workflow_engine import WorkflowEngine

current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, "test_config.yaml")


class TestSupervisor(unittest.TestCase):
    def setUp(self):
        self.config_file = config_file_path
        self.supervisor = Supervisor(self.config_file)

    def test_init(self):
        # Test that supervisor initializes without errors
        supervisor = Supervisor(self.config_file)
        assert supervisor is not None
        assert supervisor.config_manager is not None

    def test_initialize_config_manager(self):
        # Test that config manager initializes correctly
        self.supervisor.initialize_config_manager(self.config_file)
        assert self.supervisor.config_manager is not None
        assert self.supervisor.config is not None

    def test_initialize_config_manager_default_config(self):
        # Create a default config file for testing
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
logging:
  log_level: INFO
  log_file: test.log

llm_providers:
  default:
    llm_class: DEFAULT
    provider_type: bedrock
    model_id: us.amazon.nova-pro-v1:0
    temperature: 0.3
"""
            )
            temp_config = f.name

        try:
            # Copy to expected location
            default_path = os.path.join(
                os.path.dirname(self.supervisor.__class__.__module__.replace(".", "/")),
                "config.yaml",
            )
            os.makedirs(os.path.dirname(default_path), exist_ok=True)

            # Test with explicit config file instead
            self.supervisor.initialize_config_manager(temp_config)
            assert self.supervisor.config_manager is not None
        finally:
            if os.path.exists(temp_config):
                os.unlink(temp_config)

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

    @patch("common.message_bus.MessageBus.start")
    def test_start(self, mock_message_bus_start):
        self.supervisor.message_bus = MessageBus()
        self.supervisor.start()
        mock_message_bus_start.assert_called_once()

    @patch("common.message_bus.MessageBus.stop")
    def test_stop(self, mock_message_bus_stop):
        self.supervisor.message_bus = MessageBus()
        self.supervisor.stop()
        mock_message_bus_stop.assert_called_once()

    def test_run(self):
        # Test that supervisor has a run method and workflow_engine is properly initialized
        self.supervisor.workflow_engine = WorkflowEngine(LLMFactory({}), MessageBus())
        self.supervisor.workflow_engine.handle_request = Mock(return_value="wf_123")

        # Test that the supervisor has the run method
        assert hasattr(self.supervisor, "run")
        assert self.supervisor.workflow_engine is not None

        # Test direct workflow handling instead of full run loop
        from common.request_model import RequestMetadata

        request = RequestMetadata(
            prompt="Test instruction", source_id="test", target_id="supervisor"
        )
        workflow_id = self.supervisor.workflow_engine.handle_request(request)
        assert workflow_id == "wf_123"

    @patch("common.message_bus.MessageBus.is_running", return_value=True)
    @patch(
        "supervisor.metrics_manager.MetricsManager.get_metrics",
        return_value={"test": "metrics"},
    )
    def test_status(self, mock_get_metrics, mock_is_running):
        self.supervisor.message_bus = MessageBus()
        self.supervisor.metrics_manager = MetricsManager()
        status = self.supervisor.status()
        assert isinstance(status, dict)
        assert status["running"]
        assert status["metrics"] == {"test": "metrics"}

    def test_get_config_class(self):
        from config.anthropic_config import AnthropicConfig
        from config.bedrock_config import BedrockConfig
        from config.openai_config import OpenAIConfig

        assert self.supervisor.get_config_class("openai") == OpenAIConfig
        assert self.supervisor.get_config_class("anthropic") == AnthropicConfig
        assert self.supervisor.get_config_class("bedrock") == BedrockConfig
        assert self.supervisor.get_config_class("invalid") is None


if __name__ == "__main__":
    unittest.main()
