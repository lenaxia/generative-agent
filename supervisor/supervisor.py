import logging
from typing import Optional, List
from pathlib import Path
from supervisor.request_manager import RequestManager
from supervisor.agent_manager import AgentManager
from supervisor.metrics_manager import MetricsManager
from supervisor.config_manager import ConfigManager
from supervisor.logging_config import configure_logging
from shared_tools.message_bus import MessageBus, MessageType
from supervisor.task_models import Agent
from supervisor.task_graph import TaskGraph
from supervisor.llm_registry import LLMType
from supervisor.supervisor_config import SupervisorConfig
from config.bedrock_config import BedrockConfig
from config.anthropic_config import AnthropicConfig
from config.openai_config import OpenAIConfig
from llm_provider.base_client import BaseLLMClient, BedrockLLMClient, OpenAILLMClient, AnthropicLLMClient
from pydantic import BaseModel

logger = logging.getLogger("supervisor")

class Supervisor:
    config_file: Optional[str] = None
    config_manager: Optional[ConfigManager] = None
    config: Optional[SupervisorConfig] = None
    message_bus: Optional[MessageBus] = None
    request_manager: Optional[RequestManager] = None
    agent_manager: Optional[AgentManager] = None
    metrics_manager: Optional[MetricsManager] = None

    def __init__(self, config_file: Optional[str] = None):
        logger.info("Initializing Supervisor...")
        self.config_file = config_file
        self.initialize_config_manager(config_file)
        self.initialize_components()
        logger.info("Supervisor initialization complete.")

    def initialize_config_manager(self, config_file: Optional[str] = None):
        logger.info("Initializing config manager...")
        if config_file:
            self.config_manager = ConfigManager(config_file)
            logger.info(f"Using provided config file: {config_file}")
        else:
            default_config_file = Path(__file__).parent / "config.yaml"
            if default_config_file.exists():
                self.config_manager = ConfigManager(str(default_config_file))
                logger.info(f"Using default config file: {default_config_file}")
            else:
                logger.error(f"Default configuration file not found: {default_config_file}")
                raise FileNotFoundError(f"Default configuration file not found: {default_config_file}")

        logger.info("Loading config...")
        self.config = self.config_manager.load_config()
        logger.info("Config loaded successfully.")

    def initialize_components(self):
        logger.info("Initializing components...")
        configure_logging(self.config.log_level, self.config.log_file)
        logger.info(f"Logging configured with level: {self.config.log_level} and file: {self.config.log_file}")

        self.message_bus = MessageBus()
        logger.info("Message bus initialized.")

        logger.info("Populating LLM registry...")
        self.populate_llm_registry(self.config.llm_providers)

        self.request_manager = RequestManager(self.config)
        logger.info("Request manager initialized.")

        self.agent_manager = AgentManager(self.config, self.message_bus)
        logger.info("Agent manager initialized.")

        self.metrics_manager = MetricsManager(self.config)
        logger.info("Metrics manager initialized.")

        self.message_bus.subscribe(self, MessageType.TASK_RESPONSE, self.request_manager.handle_task_response)
        logger.info("Subscribed to TASK_RESPONSE messages.")

        self.message_bus.subscribe(self, MessageType.AGENT_ERROR, self.request_manager.handle_agent_error)
        logger.info("Subscribed to AGENT_ERROR messages.")

        logger.info("Component initialization complete.")

    def start(self):
        try:
            logger.info("Starting Supervisor...")
            self.agent_manager.register_agents()
            logger.info("Agents registered.")

            self.message_bus.start()
            logger.info("Message bus started.")

            logger.info("Supervisor started successfully.")
        except Exception as e:
            logger.error(f"Error starting Supervisor: {e}")

    def stop(self):
        try:
            logger.info("Stopping Supervisor...")
            self.message_bus.stop()
            logger.info("Message bus stopped.")
            logger.info("Supervisor stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping Supervisor: {e}")

    def run(self):
        try:
            logger.info("Running Supervisor...")
            self.start()
            while True:
                action = input("Enter action (status, stop): ").strip().lower()
                if action == "status":
                    status = self.status()
                    if status:
                        logger.info(f"Supervisor Status: {status}")
                    else:
                        logger.warning("Failed to retrieve Supervisor status.")
                elif action == "stop":
                    self.stop()
                    break
                else:
                    logger.warning("Invalid action. Please try again.")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping Supervisor...")
            self.stop()
        except Exception as e:
            logger.error(f"Error running Supervisor: {e}")
            self.stop()

    def status(self) -> Optional[dict]:
        try:
            status = {
                "running": self.message_bus.is_running(),
                "requests": self.request_manager.get_request_status(),
                "metrics": self.metrics_manager.get_metrics(),
            }
            logger.info(f"Retrieved Supervisor status: {status}")
            return status
        except Exception as e:
            logger.error(f"Error getting Supervisor status: {e}")
            return None

    def create_task_graph(self, instruction: str, agents: List[Agent]) -> TaskGraph:
        logger.info(f"Creating task graph for instruction: {instruction}")
        planning_agent = self.agent_manager.get_agent("PlanningAgent")
        if planning_agent:
            task_graph = planning_agent.run(instruction, agents=agents)
            logger.info("Task graph created successfully.")
            return task_graph
        else:
            logger.error("Planning Agent not found in the agent registry.")
            return None

    def populate_llm_registry(self, llm_providers: dict[str, dict]):
        logger.info("Populating LLM registry...")
        for provider_name, provider_config in llm_providers.items():
            logger.info(f"Processing LLM provider '{provider_name}'...")
            provider_type = provider_config.pop("type", None)
            if provider_type is None:
                logger.warning(f"Skipping LLM provider '{provider_name}' as the type is not specified.")
                continue

            registry_type = LLMType[provider_config.pop("registry_type", "DEFAULT").upper()]
            logger.info(f"LLM provider '{provider_name}' will be registered as type '{registry_type}'.")

            logging.info(f"Creating config for provider '{provider_name}'")
            logging.info(f"Config: {provider_config}")

            if provider_type == "openai":
                config = OpenAIConfig(**provider_config)
                llm_client = OpenAILLMClient(config, provider_name)
                logger.info(f"Created OpenAI LLM client for provider '{provider_name}'.")
            elif provider_type == "anthropic":
                config = AnthropicConfig(**provider_config)
                llm_client = AnthropicLLMClient(config, provider_name)
                logger.info(f"Created Anthropic LLM client for provider '{provider_name}'.")
            elif provider_type == "bedrock":
                config = BedrockConfig(**provider_config)
                llm_client = BedrockLLMClient(config, provider_name)
                logger.info(f"Created Bedrock LLM client for provider '{provider_name}'.")
            else:
                logger.warning(f"Unsupported LLM provider type '{provider_type}' for '{provider_name}'.")
                continue

            logging.info(f"Registering LLM client '{provider_name}' of type '{registry_type}'")
            self.config.llm_registry.register_client(llm_client, registry_type)
            logger.info(f"LLM client '{provider_name}' registered successfully.")

        logger.info("LLM registry population complete.")

if __name__ == "__main__":
    logger.info("Starting Supervisor application...")
    config_file = "config.yaml"
    supervisor = Supervisor(config_file)
    supervisor.run()
    logger.info("Supervisor application stopped.")
