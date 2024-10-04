import logging
from typing import Optional, List
from request_manager import RequestManager
from agent_manager import AgentManager
from metrics_manager import MetricsManager
from config_manager import ConfigManager
from logging_config import configure_logging
from message_bus import MessageBus, MessageType
from task_models import Agent, TaskGraph

logger = logging.getLogger(__name__)

class Supervisor:
    def __init__(self, config_file=None):
        if config_file:
            self.config_manager = ConfigManager(config_file)
        else:
            # Load default configuration file
            default_config_file = Path(__file__).parent / "config.yaml"
            if default_config_file.exists():
                self.config_manager = ConfigManager(str(default_config_file))
            else:
                raise FileNotFoundError(f"Default configuration file not found: {default_config_file}")
        self.config = self.config_manager.load_config()

        configure_logging(self.config.log_level, self.config.log_file)

        self.message_bus = MessageBus()

        self.llm_registry = LLMRegistry()
        self.populate_llm_registry(self.config.llm_providers)

        self.request_manager = RequestManager(self.config)
        self.agent_manager = AgentManager(self.config, self.message_bus)
        self.metrics_manager = MetricsManager(self.config)

        self.message_bus.subscribe(self, MessageType.TASK_RESPONSE, self.request_manager.handle_task_response)
        self.message_bus.subscribe(self, MessageType.AGENT_ERROR, self.request_manager.handle_agent_error)

    def start(self):
        try:
            self.agent_manager.register_agents()
            self.message_bus.start()
            logger.info("Supervisor started successfully.")
        except Exception as e:
            logger.error(f"Error starting Supervisor: {e}")

    def stop(self):
        try:
            self.message_bus.stop()
            logger.info("Supervisor stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping Supervisor: {e}")

    def run(self):
        try:
            self.start()
            while True:
                action = input("Enter action (status, stop): ").strip().lower()
                if action == "status":
                    status = self.status()
                    print(f"Supervisor Status: {status}")
                elif action == "stop":
                    self.stop()
                    break
                else:
                    print("Invalid action. Please try again.")
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logger.error(f"Error running Supervisor: {e}")
            self.stop()

    def status(self) -> Optional[dict]:
        try:
            return {
                "running": self.message_bus.is_running(),
                "requests": self.request_manager.get_request_status(),
                "metrics": self.metrics_manager.get_metrics(),
            }
        except Exception as e:
            logger.error(f"Error getting Supervisor status: {e}")
            return None

    def create_task_graph(self, instruction: str, agents: List[Agent]) -> TaskGraph:
        planning_agent = self.agent_manager.get_agent("PlanningAgent")
        if planning_agent:
            task_graph = planning_agent.run(instruction, agents=agents)
            return task_graph
        else:
            logger.error("Planning Agent not found in the agent registry.")
            return None

    def populate_llm_registry(self, llm_providers: Dict[str, Dict]):
        for provider_name, provider_config in llm_providers.items():
            provider_type = provider_config.pop("type", None)
            if provider_type is None:
                logger.warning(f"Skipping LLM provider '{provider_name}' as the type is not specified.")
                continue

            if provider_type == "openai":
                config = OpenAIConfig(**provider_config)
                llm_client = OpenAILLMClient(config, provider_name)
            elif provider_type == "anthropic":
                config = AnthropicConfig(**provider_config)
                llm_client = AnthropicLLMClient(config, provider_name)
            elif provider_type == "bedrock":
                config = ChatBedrockConfig(**provider_config)
                llm_client = BedrockLLMClient(config, provider_name)
            else:
                logger.warning(f"Unsupported LLM provider type '{provider_type}' for '{provider_name}'.")
                continue

            self.llm_registry.register_client(llm_client)

if __name__ == "__main__":
    supervisor = Supervisor("config.yaml")
    supervisor.run()
