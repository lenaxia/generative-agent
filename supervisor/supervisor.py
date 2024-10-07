import logging
import sys
import time
from typing import Optional, List
from pathlib import Path
from supervisor.request_manager import RequestManager, RequestModel
from supervisor.agent_manager import AgentManager
from supervisor.metrics_manager import MetricsManager
from supervisor.config_manager import ConfigManager
from supervisor.logging_config import configure_logging
from shared_tools.message_bus import MessageBus, MessageType
from supervisor.supervisor_config import SupervisorConfig
from config.bedrock_config import BedrockConfig
from config.anthropic_config import AnthropicConfig
from config.openai_config import OpenAIConfig
from llm_provider.factory import LLMFactory, LLMType

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

        logger.info("Initializing LLM factory...")
        llm_factory = LLMFactory({})

        # Populate the LLM factory with configurations from self.config.llm_providers
        for provider_name, provider_config in self.config.llm_providers.items():
            llm_type = LLMType[provider_config.pop("registry_type", "DEFAULT").upper()]
            config_cls = self.get_config_class(provider_config.pop("type", None))
            if config_cls:
                config = config_cls(**provider_config)
                llm_factory.add_config(llm_type, config)
            else:
                logger.warning(f"Unsupported LLM provider type for '{provider_name}'.")

        self.agent_manager = AgentManager(self.config, self.message_bus, llm_factory)
        logger.info("Agent manager initialized.")

        self.request_manager = RequestManager(self.agent_manager, self.message_bus)
        logger.info("Request manager initialized.")

        self.metrics_manager = MetricsManager()
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
        #try:
        logger.info("Running Supervisor...")
        self.start()
        while True:
            action = input("Enter action (instruction, status, stop): ").strip().lower()
            if action == "stop":
                self.stop()
                break
            elif action == "status":
                status = self.status()
                if status:
                    logger.info(f"Supervisor Status: {status}")
                else:
                    logger.warning("Failed to retrieve Supervisor status.")
            else:
                if len(action) < 5:
                    logger.warning("Invalid instruction. Please enter at least 5 characters.")
                    continue
                request = RequestModel(instructions=action)
                request_id = self.request_manager.handle_request(request)
                logger.info(f"New request '{request_id}' created and delegated.")

                request_completed = False
                while not request_completed:
                    progress_info = self.request_manager.monitor_progress(request_id)
                    if progress_info is None:
                        logger.info(f"Request '{request_id}' completed or failed.")
                        request_completed = True
                    else:
                        logger.info(f"Request '{request_id}' Status: {progress_info}")
                        if progress_info.get("graph_completed", False):
                            request_completed = True
                        else:
                            time.sleep(5)  # Wait for 5 seconds before checking progress again
        #except KeyboardInterrupt:
        #    logger.info("Keyboard interrupt received. Stopping Supervisor...")
        #    self.stop()
        #except Exception as e:
        #    logger.error(f"Error running Supervisor: {e}")
        #    sys.exit(1)

    def status(self) -> Optional[dict]:
        try:
            status = {
                "running": self.message_bus.is_running(),
                #"requests": self.request_manager.get_request_status(),
                "metrics": self.metrics_manager.get_metrics(),
            }
            logger.info(f"Retrieved Supervisor status: {status}")
            return status
        except Exception as e:
            logger.error(f"Error getting Supervisor status: {e}")
            return None

    def get_config_class(self, provider_type):
        if provider_type == "openai":
            return OpenAIConfig
        elif provider_type == "anthropic":
            return AnthropicConfig
        elif provider_type == "bedrock":
            return BedrockConfig
        else:
            return None

if __name__ == "__main__":
    logger.info("Starting Supervisor application...")
    config_file = "config.yaml"
    supervisor = Supervisor(config_file)
    supervisor.run()
    logger.info("Supervisor application stopped.")
