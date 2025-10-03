import logging
import sys
import time
from typing import Optional, List
from pathlib import Path
from supervisor.workflow_engine import WorkflowEngine
from supervisor.heartbeat import Heartbeat
from common.request_model import RequestMetadata
from supervisor.metrics_manager import MetricsManager
from supervisor.config_manager import ConfigManager
from supervisor.logging_config import configure_logging
from common.message_bus import MessageBus, MessageType
from supervisor.supervisor_config import SupervisorConfig, LLMProviderConfig
from config.bedrock_config import BedrockConfig
from config.anthropic_config import AnthropicConfig
from config.openai_config import OpenAIConfig
from llm_provider.factory import LLMFactory, LLMType

logger = logging.getLogger("supervisor")

class Supervisor:
    # TODO: It should be able to do it mostly out of the box right now, but we should confirm whether or not we can nest supervisors, to basically do
    #       teams of teams. Each team would get its own MessageBus to communicate internally, and then the Supervisor would be responsible for communicating
    #       with the other teams and the top level supervisor.
    config_file: Optional[str] = None
    config_manager: Optional[ConfigManager] = None
    config: Optional[SupervisorConfig] = None
    message_bus: Optional[MessageBus] = None
    workflow_engine: Optional[WorkflowEngine] = None
    metrics_manager: Optional[MetricsManager] = None
    llm_factory: Optional[LLMFactory] = None
    heartbeat: Optional[Heartbeat] = None

    def __init__(self, config_file: Optional[str] = None):
        """
        Initializes the Supervisor with the given configuration file.

        If no configuration file is given, it will use the default configuration
        file name.

        Args:
            config_file: The path to the configuration file.
        """
        logger.info("Initializing Supervisor...")
        self.config_file = config_file
        self.initialize_config_manager(config_file)
        self.initialize_components()
        logger.info("Supervisor initialization complete.")

    def initialize_config_manager(self, config_file: Optional[str] = None):
        """
        Initializes the config manager and loads the configuration.

        If a configuration file is provided, it will be used to initialize the
        config manager. Otherwise, the default configuration file will be used.

        Raises:
            FileNotFoundError: If the default configuration file is not found.
        """
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
        """
        Initializes all components of the supervisor.

        This includes setting up logging, initializing the message bus, populating
        the LLM factory with configurations, initializing the request manager with
        Universal Agent, initializing the task scheduler, and initializing the metrics manager.

        It also sets up subscriptions to TASK_RESPONSE and AGENT_ERROR messages.

        This function is idempotent and can be called multiple times without
        causing any issues.
        """
        logger.info("Initializing components...")
        configure_logging(self.config.logging)
        logger.info(f"Logging configured with level: {self.config.logging.log_level} and file: {self.config.logging.log_file}")

        self.message_bus = MessageBus()
        logger.info("Message bus initialized.")

        logger.info("Initializing LLM factory...")
        self.llm_factory = LLMFactory({})

        # Populate the LLM factory with configurations from self.config.llm_providers
        for provider_name, provider_config in self.config.llm_providers.items():
            # Handle the new YAML structure: llm_providers.bedrock.models.WEAK/DEFAULT/STRONG
            if 'models' in provider_config:
                # New structure: bedrock/openai with models mapping
                models = provider_config['models']
                parameters = provider_config.get('parameters', {})
                
                for llm_type_str, model_id in models.items():
                    try:
                        llm_type = LLMType[llm_type_str.upper()]
                        
                        # Create appropriate config object based on provider
                        if provider_name.lower() == 'bedrock':
                            from config.bedrock_config import BedrockConfig
                            config_obj = BedrockConfig(
                                name=f"{provider_name}_{llm_type_str}",
                                model_id=model_id,
                                region=parameters.get('region', 'us-west-2'),
                                temperature=parameters.get('temperature', 0.3),
                                max_tokens=parameters.get('max_tokens', 4096)
                            )
                        elif provider_name.lower() == 'openai':
                            from config.openai_config import OpenAIConfig
                            config_obj = OpenAIConfig(
                                name=f"{provider_name}_{llm_type_str}",
                                model_id=model_id,
                                temperature=parameters.get('temperature', 0.3),
                                max_tokens=parameters.get('max_tokens', 4096)
                            )
                        else:
                            logger.warning(f"Unknown provider type: {provider_name}")
                            continue
                            
                        self.llm_factory.add_config(llm_type, config_obj)
                        logger.info(f"Added {llm_type.value} config for {provider_name}: {model_id}")
                        
                    except KeyError:
                        logger.warning(f"Unknown LLM type: {llm_type_str}")
                        continue
            else:
                # Legacy structure: direct provider config with llm_class
                llm_class_str = provider_config.get("llm_class", "DEFAULT")
                if isinstance(llm_class_str, str):
                    llm_class = LLMType[llm_class_str.upper()]
                else:
                    llm_class = llm_class_str
                self.llm_factory.add_config(llm_class, provider_config)
            
        logger.info("LLM factory initialized with Universal Agent support.")

        # Initialize WorkflowEngine (consolidated RequestManager + TaskScheduler)
        # Pass MCP config path only if MCP integration is enabled
        mcp_config_path = None
        if (hasattr(self.config, 'mcp') and self.config.mcp and self.config.mcp.enabled) or \
           (hasattr(self.config, 'feature_flags') and self.config.feature_flags and self.config.feature_flags.enable_mcp_integration):
            mcp_config_path = self.config.mcp.config_file if hasattr(self.config, 'mcp') and self.config.mcp else 'config/mcp_config.yaml'
        
        self.workflow_engine = WorkflowEngine(
            llm_factory=self.llm_factory,
            message_bus=self.message_bus,
            max_concurrent_tasks=5,
            checkpoint_interval=300,
            mcp_config_path=mcp_config_path
        )
        logger.info("WorkflowEngine initialized (consolidated RequestManager + TaskScheduler).")

        self.metrics_manager = MetricsManager()
        logger.info("Metrics manager initialized.")

        # Initialize Heartbeat service
        heartbeat_config = getattr(self.config, 'heartbeat', {})
        heartbeat_interval = heartbeat_config.get('interval', 30) if isinstance(heartbeat_config, dict) else 30
        health_check_interval = heartbeat_config.get('health_check_interval', 60) if isinstance(heartbeat_config, dict) else 60
        
        self.heartbeat = Heartbeat(
            supervisor=self,
            interval=heartbeat_interval,
            health_check_interval=health_check_interval
        )
        logger.info(f"Heartbeat service initialized with {heartbeat_interval}s interval.")

        # Note: Message bus subscriptions are handled by WorkflowEngine internally
        logger.info("Component initialization complete.")

    def start(self):
        """
        Starts the Supervisor by starting the message bus and task scheduler.

        This method can be invoked multiple times without causing any issues.
        """
        try:
            logger.info("Starting Supervisor...")
            self.message_bus.start()
            logger.info("Message bus started.")
            
            self.workflow_engine.start_workflow_engine()
            logger.info("WorkflowEngine started.")
            
            # Start heartbeat service
            if self.heartbeat:
                self.heartbeat.start()
                logger.info("Heartbeat service started.")

            logger.info("Supervisor started successfully.")
        except Exception as e:
            logger.error(f"Error starting Supervisor: {e}")

    def stop(self):
        """
        Stops the Supervisor by stopping the task scheduler and message bus.

        This method can be invoked multiple times without causing any issues.
        """
        try:
            logger.info("Stopping Supervisor...")
            
            # Stop heartbeat service first
            if self.heartbeat:
                self.heartbeat.stop()
                logger.info("Heartbeat service stopped.")
            
            self.workflow_engine.stop_workflow_engine()
            logger.info("WorkflowEngine stopped.")
            
            self.message_bus.stop()
            logger.info("Message bus stopped.")
            logger.info("Supervisor stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping Supervisor: {e}")

    def run(self):
        """
        Runs the Supervisor with the WorkflowEngine and Universal Agent.
        Enters an interactive loop to process user workflows.

        The user can enter one of the following commands:

        *   workflow: A workflow instruction that will be executed by the Universal Agent.
        *   status: Retrieves the Supervisor and WorkflowEngine status.
        *   stop: Stops the Supervisor and exits the program.

        The Supervisor will display workflow progress and notify when workflows complete or fail.
        """
        try:
            logger.info("Running Supervisor...")
            self.start()
            while True:
                action = input("Enter action (workflow, status, stop): ").strip().lower()
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
                    request = RequestMetadata(
                        prompt=action,
                        source_id="console",
                        target_id="supervisor",
                    )
                    workflow_id = self.workflow_engine.start_workflow(action)
                    logger.info(f"New workflow '{workflow_id}' started.")

                    workflow_completed = False
                    while not workflow_completed:
                        progress_info = self.workflow_engine.get_request_status(workflow_id)
                        if progress_info is None:
                            workflow_completed = True
                        else:
                            logger.info(f"Workflow '{workflow_id}' Status: {progress_info}")
                            if progress_info.get("status", False):
                                workflow_completed = True
                            else:
                                time.sleep(5)  # Wait for 5 seconds before checking progress again
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping Supervisor...")
            self.stop()
        except Exception as e:
            logger.error(f"Error running Supervisor: {e}")
            sys.exit(1)

    def status(self) -> Optional[dict]:
        """
        Retrieves the current status of the Supervisor, including whether it is running,
        the current metrics, and the status of all requests.

        Returns:
            Optional[dict]: The Supervisor status, or None if an error occurred.
        """
        try:
            status = {
                "status": "running" if self.message_bus.is_running() else "stopped",
                "running": self.message_bus.is_running(),
                "workflow_engine": self.workflow_engine.get_workflow_metrics() if self.workflow_engine else None,
                "universal_agent": self.workflow_engine.get_universal_agent_status() if self.workflow_engine else None,
                "heartbeat": self.heartbeat.get_health_status() if self.heartbeat else None,
                "metrics": self.metrics_manager.get_metrics(),
            }
            logger.info(f"Retrieved Supervisor status: {status}")
            return status
        except Exception as e:
            logger.error(f"Error getting Supervisor status: {e}")
            return None

    def get_config_class(self, provider_type):
        # TODO: [Low] We need to get rid of this method and extract the type dynamically, we don't want to be tied to hard coded definitions
        
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
