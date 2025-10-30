"""Main supervisor module for workflow orchestration and management.

This module provides the core Supervisor class that orchestrates workflows,
manages agent execution, coordinates between different components, and
ensures proper system operation and monitoring.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from common.intent_processor import IntentProcessor
from common.intents import WorkflowIntent
from common.message_bus import MessageBus, MessageType
from config.anthropic_config import AnthropicConfig
from config.bedrock_config import BedrockConfig
from config.openai_config import OpenAIConfig
from llm_provider.factory import LLMFactory, LLMType
from supervisor.config_manager import ConfigManager
from supervisor.logging_config import configure_logging
from supervisor.metrics_manager import MetricsManager
from supervisor.supervisor_config import SupervisorConfig
from supervisor.workflow_engine import WorkflowEngine

logger = logging.getLogger("supervisor")


class Supervisor:
    """Central coordination and management system for the StrandsAgent architecture.

    Manages workflow execution, agent coordination, message routing, and system
    monitoring. Supports hierarchical supervision patterns and team-based
    organization for scalable multi-agent systems.

    Note: Future enhancement planned for nested supervisor support to enable
    teams of teams architecture with dedicated MessageBus per team.
    """

    # TODO: It should be able to do it mostly out of the box right now, but we should confirm whether or not we can nest supervisors, to basically do
    #       teams of teams. Each team would get its own MessageBus to communicate internally, and then the Supervisor would be responsible for communicating
    #       with the other teams and the top level supervisor.
    config_file: str | None = None
    config_manager: ConfigManager | None = None
    config: SupervisorConfig | None = None
    message_bus: MessageBus | None = None
    workflow_engine: WorkflowEngine | None = None
    metrics_manager: MetricsManager | None = None
    llm_factory: LLMFactory | None = None
    communication_manager: object | None = None  # Import will be done in method
    intent_processor: IntentProcessor | None = None
    suspended_requests: dict = None

    def __init__(self, config_file: str | None = None):
        """Initializes the Supervisor with the given configuration file.

        If no configuration file is given, it will use the default configuration
        file name.

        Args:
            config_file: The path to the configuration file.
        """
        logger.info("Initializing LLM-safe Supervisor...")
        self.config_file = config_file

        self._scheduled_tasks: list[dict] = []
        self._scheduled_intervals: dict[str, float] = {}

        # Initialize intent processing and workflow suspension
        self.intent_processor = None
        self.suspended_requests = {}

        # Heartbeat failure tracking
        self._heartbeat_failure_count = 0
        self._heartbeat_max_failures = 3

        self.initialize_config_manager(config_file)
        self._set_environment_variables()
        self.initialize_components()

        # Schedule Bedrock connection heartbeat to prevent idle timeouts
        self._schedule_bedrock_heartbeat()

        logger.info("LLM-safe Supervisor initialization complete.")

    def add_scheduled_task(self, task: dict) -> None:
        """Add task to scheduled execution queue.

        Args:
            task: Task dictionary with 'type', 'handler', and optional 'interval', 'intent', 'data'
        """
        self._scheduled_tasks.append(task)
        logger.debug(f"Added scheduled task: {task.get('type', 'unknown')}")

    def process_scheduled_tasks(self) -> None:
        """Process scheduled tasks in single event loop."""
        import time

        current_time = time.time()

        # Process one-time tasks
        one_time_tasks = [
            task for task in self._scheduled_tasks if "interval" not in task
        ]
        for task in one_time_tasks:
            try:
                handler = task.get("handler")
                if callable(handler):
                    if "intent" in task:
                        # Process intent-based tasks
                        handler(task["intent"])
                    elif "data" in task:
                        # Process tasks with data
                        handler(task["data"])
                    else:
                        # Process regular tasks
                        handler(task)

                self._scheduled_tasks.remove(task)
                logger.debug(f"Executed scheduled task: {task.get('type', 'unknown')}")

            except Exception as e:
                logger.error(f"Scheduled task failed: {e}")
                self._scheduled_tasks.remove(task)  # Remove failed tasks

        # Process interval tasks
        interval_tasks = [task for task in self._scheduled_tasks if "interval" in task]
        for task in interval_tasks:
            task_type = task.get("type", "unknown")
            interval = task.get("interval", 60)
            last_run = self._scheduled_intervals.get(task_type, 0)

            if current_time - last_run >= interval:
                try:
                    handler = task.get("handler")
                    if callable(handler):
                        handler()

                    self._scheduled_intervals[task_type] = current_time
                    logger.debug(f"Executed interval task: {task_type}")

                except Exception as e:
                    logger.error(f"Interval task {task_type} failed: {e}")

    def _schedule_bedrock_heartbeat(self):
        """Schedule recurring heartbeat to keep Bedrock connection alive.

        This prevents idle connection timeouts by periodically validating
        AWS credentials, which keeps the boto3 connection pool active.
        Uses FREE AWS STS API call - no Bedrock/LLM costs incurred.
        """
        heartbeat_task = {
            "type": "bedrock_heartbeat",
            "handler": self._bedrock_heartbeat,
            "interval": 300,  # 5 minutes (300 seconds)
        }

        self.add_scheduled_task(heartbeat_task)
        logger.info("📡 Bedrock heartbeat scheduled (5-minute interval, zero cost)")

    def _bedrock_heartbeat(self):
        """Keep Bedrock runtime connection alive with minimal inference call.

        This method is called every 5 minutes to prevent connection pool
        timeouts. It uses bedrock-runtime with a minimal 1-token inference
        to keep the actual runtime connection pool active.

        Cost: ~$0.00025 per call = $0.072/day = $2.16/month
        Benefit: Prevents 123s idle delays, achieves 0-2s wake-up time

        Tracks failures and attempts recovery after sustained failures.
        """
        try:
            # Use bedrock-runtime (same service Strands uses for inference)
            import json

            import boto3

            bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-west-2")

            # Make minimal inference call with cheapest/fastest model
            # Use Haiku with max_tokens=1 to minimize cost
            body = json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1,  # Minimal output
                    "messages": [{"role": "user", "content": "hi"}],
                }
            )

            bedrock_runtime.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0", body=body
            )

            # Reset failure count on success
            if self._heartbeat_failure_count > 0:
                logger.info(
                    f"✅ Bedrock heartbeat recovered after {self._heartbeat_failure_count} failures"
                )
                self._heartbeat_failure_count = 0

            logger.debug("✅ Bedrock runtime heartbeat successful (~$0.00025)")

        except Exception as e:
            self._heartbeat_failure_count += 1
            logger.warning(
                f"⚠️ Bedrock heartbeat failed ({self._heartbeat_failure_count}/{self._heartbeat_max_failures}): {e}"
            )

            # Attempt recovery after sustained failures
            if self._heartbeat_failure_count >= self._heartbeat_max_failures:
                logger.error(
                    f"🚨 Bedrock heartbeat failed {self._heartbeat_max_failures} times, attempting recovery..."
                )
                self._attempt_heartbeat_recovery()

    def _attempt_heartbeat_recovery(self):
        """Attempt to recover from sustained heartbeat failures.

        Note: Since Strands BedrockModel doesn't support custom boto3 clients,
        we can only log the issue and reset the counter. The heartbeat will
        continue attempting, and boto3's internal retry logic will handle recovery.
        """
        try:
            logger.info("🔄 Attempting heartbeat recovery...")

            # Clear model cache to force recreation on next use
            if self.llm_factory:
                self.llm_factory.clear_cache()
                logger.info("✅ Cleared model cache to force fresh connections")

            # Reset failure count after recovery attempt
            self._heartbeat_failure_count = 0
            logger.info("✅ Heartbeat recovery completed")

        except Exception as e:
            logger.error(f"❌ Failed to recover from heartbeat failures: {e}")
            # Keep trying on next heartbeat

    def initialize_config_manager(self, config_file: str | None = None):
        """Initializes the config manager and loads the configuration.

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
                logger.error(
                    f"Default configuration file not found: {default_config_file}"
                )
                raise FileNotFoundError(
                    f"Default configuration file not found: {default_config_file}"
                )

        logger.info("Loading config...")
        self.config = self.config_manager.load_config()
        logger.info("Config loaded successfully.")

    def _set_environment_variables(self):
        """Set environment variables from configuration."""
        if not self.config_manager:
            return

        # Access the raw config data from config manager
        raw_config = getattr(self.config_manager, "raw_config_data", {})
        env_config = raw_config.get("environment", {})
        if env_config:
            logger.info("Setting environment variables from configuration...")
            for key, value in env_config.items():
                os.environ[key] = str(value)
                logger.debug(f"Set environment variable: {key}={value}")

    def initialize_components(self):
        """Initializes all components of the supervisor.

        This includes setting up logging, initializing the message bus, populating
        the LLM factory with configurations, initializing the request manager with
        Universal Agent, initializing the task scheduler, and initializing the metrics manager.

        It also sets up subscriptions to TASK_RESPONSE and AGENT_ERROR messages.

        This function is idempotent and can be called multiple times without
        causing any issues.
        """
        logger.info("Initializing components...")

        self._initialize_logging()
        self._initialize_message_bus()
        self._initialize_communication_manager()
        self._initialize_llm_factory()
        self._initialize_workflow_engine()
        self._initialize_metrics_manager()
        self._initialize_intent_processor()

        logger.info("Component initialization complete.")

    def _initialize_logging(self):
        """Initialize logging configuration."""
        configure_logging(self.config.logging)
        logger.info(
            f"Logging configured with level: {self.config.logging.log_level} and file: {self.config.logging.log_file}"
        )

    def _initialize_message_bus(self):
        """Initialize message bus."""
        self.message_bus = MessageBus()
        logger.info("Message bus initialized.")

    def _initialize_communication_manager(self):
        """Initialize communication manager with supervisor's MessageBus."""
        from common.communication_manager import CommunicationManager

        self.communication_manager = CommunicationManager(
            self.message_bus, supervisor=self
        )
        # Initialize channels synchronously during startup
        self.communication_manager.initialize_sync()

        # Start the async queue processor after supervisor starts
        # This will be handled in the start() method
        logger.info("Communication manager initialized with channel handlers")

    def _initialize_llm_factory(self):
        """Initialize LLM factory with provider configurations."""
        logger.info("Initializing LLM factory...")
        self.llm_factory = LLMFactory({})

        # Populate the LLM factory with configurations from self.config.llm_providers
        for provider_name, provider_config in self.config.llm_providers.items():
            self._process_provider_config(provider_name, provider_config)

        logger.info("LLM factory initialized with Universal Agent support.")

    def _process_provider_config(self, provider_name: str, provider_config: dict):
        """Process a single provider configuration."""
        if "models" in provider_config:
            self._process_new_provider_structure(provider_name, provider_config)
        else:
            self._process_legacy_provider_structure(provider_config)

    def _process_new_provider_structure(
        self, provider_name: str, provider_config: dict
    ):
        """Process new YAML structure: llm_providers.bedrock.models.WEAK/DEFAULT/STRONG."""
        models = provider_config["models"]
        parameters = provider_config.get("parameters", {})

        for llm_type_str, model_id in models.items():
            try:
                llm_type = LLMType[llm_type_str.upper()]
                config_obj = self._create_provider_config(
                    provider_name, llm_type_str, model_id, parameters
                )

                if config_obj:
                    self.llm_factory.add_config(llm_type, config_obj)
                    logger.info(
                        f"Added {llm_type.value} config for {provider_name}: {model_id}"
                    )

            except KeyError:
                logger.warning(f"Unknown LLM type: {llm_type_str}")
                continue

    def _create_provider_config(
        self, provider_name: str, llm_type_str: str, model_id: str, parameters: dict
    ):
        """Create appropriate config object based on provider."""
        if provider_name.lower() == "bedrock":
            from config.bedrock_config import BedrockConfig

            return BedrockConfig(
                name=f"{provider_name}_{llm_type_str}",
                model=model_id,  # BedrockModelConfig expects 'model', not 'model_id'
                region=parameters.get("region", "us-west-2"),
                temperature=parameters.get("temperature", 0.3),
                max_tokens=parameters.get("max_tokens", 4096),
            )
        elif provider_name.lower() == "openai":
            from config.openai_config import OpenAIConfig

            return OpenAIConfig(
                name=f"{provider_name}_{llm_type_str}",
                model_id=model_id,
                temperature=parameters.get("temperature", 0.3),
                max_tokens=parameters.get("max_tokens", 4096),
            )
        else:
            logger.warning(f"Unknown provider type: {provider_name}")
            return None

    def _process_legacy_provider_structure(self, provider_config: dict):
        """Process legacy structure: direct provider config with llm_class."""
        llm_class_str = provider_config.get("llm_class", "DEFAULT")
        if isinstance(llm_class_str, str):
            llm_class = LLMType[llm_class_str.upper()]
        else:
            llm_class = llm_class_str
        self.llm_factory.add_config(llm_class, provider_config)

    def _initialize_workflow_engine(self):
        """Initialize WorkflowEngine with MCP and fast-path configurations."""
        mcp_config_path = self._get_mcp_config_path()
        fast_path_config = self._get_fast_path_config()

        self.workflow_engine = WorkflowEngine(
            llm_factory=self.llm_factory,
            message_bus=self.message_bus,
            max_concurrent_tasks=5,
            checkpoint_interval=300,
            mcp_config_path=mcp_config_path,
            fast_path_config=fast_path_config,
        )

        # Inject communication_manager into MessageBus for event handler dependencies
        if hasattr(self, "communication_manager") and self.communication_manager:
            self.message_bus.communication_manager = self.communication_manager

        logger.info(
            "WorkflowEngine initialized (consolidated RequestManager + TaskScheduler)."
        )

    def _get_mcp_config_path(self):
        """Get MCP config path if MCP integration is enabled."""
        if (
            hasattr(self.config, "mcp") and self.config.mcp and self.config.mcp.enabled
        ) or (
            hasattr(self.config, "feature_flags")
            and self.config.feature_flags
            and self.config.feature_flags.enable_mcp_integration
        ):
            return (
                self.config.mcp.config_file
                if hasattr(self.config, "mcp") and self.config.mcp
                else "config/mcp_config.yaml"
            )
        return None

    def _get_fast_path_config(self):
        """Get fast-path configuration from config."""
        if not (hasattr(self.config, "fast_path") and self.config.fast_path):
            return None

        return {
            "enabled": getattr(self.config.fast_path, "enabled", True),
            "confidence_threshold": getattr(
                self.config.fast_path, "confidence_threshold", 0.7
            ),
            "max_response_time": getattr(
                self.config.fast_path, "max_response_time", 3000
            ),
            "fallback_on_error": getattr(
                self.config.fast_path, "fallback_on_error", True
            ),
            "log_routing_decisions": getattr(
                self.config.fast_path, "log_routing_decisions", True
            ),
            "track_performance_metrics": getattr(
                self.config.fast_path, "track_performance_metrics", True
            ),
        }

    def _initialize_metrics_manager(self):
        """Initialize metrics manager."""
        self.metrics_manager = MetricsManager()
        logger.info("Metrics manager initialized.")

    def _initialize_intent_processor(self):
        """Initialize intent processor and register workflow intent handlers."""
        self.intent_processor = IntentProcessor(
            communication_manager=self.communication_manager,
            workflow_engine=self.workflow_engine,
            message_bus=self.message_bus,
        )

        # Register workflow intent handler
        self.intent_processor.register_role_intent_handler(
            WorkflowIntent, self.handle_workflow_execution_intent, "supervisor"
        )

        # Subscribe to workflow completion events
        self.message_bus.subscribe(
            self, "WORKFLOW_COMPLETED", self.handle_workflow_completed
        )

        # Document 35 Phase 2: Inject intent processor into Universal Agent
        if hasattr(self.workflow_engine, "universal_agent"):
            self.workflow_engine.universal_agent.intent_processor = (
                self.intent_processor
            )
            logger.info(
                "Intent processor injected into Universal Agent for Phase 2 intent detection"
            )

        # Also inject into role registry for backward compatibility
        if hasattr(self.workflow_engine, "role_registry"):
            self.workflow_engine.role_registry.set_intent_processor(
                self.intent_processor
            )

        logger.info("Intent processor initialized with workflow intent handlers.")

    def handle_workflow_execution_intent(self, intent: WorkflowIntent):
        """Handle workflow execution intent by suspending request and starting execution."""
        # Suspend original request
        self.suspended_requests[intent.request_id] = {
            "intent": intent,
            "suspended_at": time.time(),
            "user_id": intent.user_id,
            "channel_id": intent.channel_id,
        }

        # Start workflow execution
        workflow_id = self.workflow_engine.execute_workflow_intent(intent)

        logger.info(f"Workflow {workflow_id} started for request {intent.request_id}")
        return f"Multi-step workflow initiated"

    def handle_workflow_completed(self, event_data):
        """Resume suspended request with consolidated results."""
        request_id = event_data["request_id"]

        if request_id in self.suspended_requests:
            suspended_request = self.suspended_requests[request_id]

            # Send results back to user via message bus (LLM-SAFE: single event loop)
            self.message_bus.publish(
                self,
                MessageType.SEND_MESSAGE,
                {
                    "message": event_data["consolidated_results"],
                    "context": {
                        "channel_id": suspended_request["channel_id"],
                        "user_id": suspended_request["user_id"],
                        "request_id": request_id,
                    },
                },
            )

            # Clean up suspended request
            del self.suspended_requests[request_id]

            logger.info(f"Workflow results sent for request {request_id}")

    def start(self):
        """Starts the Supervisor by starting the message bus and task scheduler.

        This method can be invoked multiple times without causing any issues.
        """
        try:
            logger.info("Starting Supervisor...")
            self.message_bus.start()
            logger.info("Message bus started.")

            self.workflow_engine.start_workflow_engine()
            logger.info("WorkflowEngine started.")

            # Defer scheduled tasks to async context
            try:
                self._start_scheduled_tasks()
                logger.info(
                    "Scheduled heartbeat tasks started (single event loop mode)"
                )
            except RuntimeError as e:
                logger.warning(f"Heartbeat tasks deferred - no event loop: {e}")
                logger.info(
                    "Heartbeat tasks will start when async context is available"
                )

            logger.info("Supervisor started successfully.")
        except Exception as e:
            logger.error(f"Error starting Supervisor: {e}")

    async def start_async_tasks(self):
        """Start async tasks that require an event loop context."""
        if hasattr(self, "_async_tasks_started"):
            logger.debug(f"_async_tasks_started = {self._async_tasks_started}")

        if not hasattr(self, "_async_tasks_started") or not self._async_tasks_started:
            logger.debug("First time calling start_async_tasks - proceeding")
            try:
                self._start_scheduled_tasks()
                logger.info("Async heartbeat tasks started successfully")

                # Initialize context systems if workflow engine supports it
                if hasattr(self.workflow_engine, "initialize_context_systems"):
                    try:
                        await self.workflow_engine.initialize_context_systems()
                        logger.info("Context systems initialized successfully")
                    except Exception as e:
                        logger.warning(f"Context systems initialization failed: {e}")
                        # System continues without context awareness

                self._async_tasks_started = True
                logger.debug(f"Set _async_tasks_started = {self._async_tasks_started}")
            except Exception as e:
                logger.error(f"Failed to start async tasks: {e}")
                raise
        else:
            logger.debug("start_async_tasks already called - skipping")

    def stop(self):
        """Stops the Supervisor by stopping the task scheduler and message bus.

        This method can be invoked multiple times without causing any issues.
        """
        try:
            logger.info("Stopping Supervisor...")

            # Stop communication manager first to cleanup channel handlers
            if self.communication_manager:
                import asyncio

                try:
                    # Run the async shutdown method
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're in an async context, create a task
                        asyncio.create_task(self.communication_manager.shutdown())
                    else:
                        # If we're in a sync context, run until complete
                        loop.run_until_complete(self.communication_manager.shutdown())
                except RuntimeError:
                    # No event loop, create a new one
                    asyncio.run(self.communication_manager.shutdown())
                except Exception as e:
                    logger.error(f"Error shutting down communication manager: {e}")

            # Stop services based on mode

            self._stop_scheduled_tasks()
            logger.info("Scheduled tasks stopped (single event loop mode)")

            self.workflow_engine.stop_workflow_engine()
            logger.info("WorkflowEngine stopped.")

            self.message_bus.stop()
            logger.info("Message bus stopped.")
            logger.info("Supervisor stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping Supervisor: {e}")

    def run(self):
        """Runs the Supervisor with the WorkflowEngine and Universal Agent.

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
            self._run_interactive_loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping Supervisor...")
            self.stop()
        except Exception as e:
            logger.error(f"Error running Supervisor: {e}")
            sys.exit(1)

    def _run_interactive_loop(self):
        """Run the main interactive loop."""
        while True:
            action = input("Enter action (workflow, status, stop): ").strip().lower()

            if action == "stop":
                self.stop()
                break
            elif action == "status":
                self._handle_status_action()
            else:
                self._handle_workflow_action(action)

    def _handle_status_action(self):
        """Handle status action."""
        status = self.status()
        if status:
            logger.info(f"Supervisor Status: {status}")
        else:
            logger.warning("Failed to retrieve Supervisor status.")

    def _handle_workflow_action(self, action: str):
        """Handle workflow action."""
        if len(action) < 5:
            logger.warning("Invalid instruction. Please enter at least 5 characters.")
            return

        workflow_id = self.workflow_engine.start_workflow(action)
        logger.info(f"New workflow '{workflow_id}' started.")
        self._monitor_workflow_progress(workflow_id)

    def _monitor_workflow_progress(self, workflow_id: str):
        """Monitor workflow progress until completion."""
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

    def status(self) -> dict | None:
        """Retrieves the current status of the Supervisor, including whether it is running,

        the current metrics, and the status of all requests.

        Returns:
            Optional[dict]: The Supervisor status, or None if an error occurred.
        """
        try:
            status = {
                "status": "running" if self.message_bus.is_running() else "stopped",
                "running": self.message_bus.is_running(),
                "workflow_engine": (
                    self.workflow_engine.get_workflow_metrics()
                    if self.workflow_engine
                    else None
                ),
                "universal_agent": (
                    self.workflow_engine.get_universal_agent_status()
                    if self.workflow_engine
                    else None
                ),
                "heartbeat": (
                    {
                        "overall_status": "healthy",
                        "components_healthy": (
                            len(self._scheduled_tasks)
                            if hasattr(self, "_scheduled_tasks")
                            else 0
                        ),
                        "mode": "scheduled_tasks",
                        "scheduled_tasks_count": (
                            len(self._scheduled_tasks)
                            if hasattr(self, "_scheduled_tasks")
                            else 0
                        ),
                    }  # LLM-SAFE: Scheduled task health
                ),
                "metrics": self.metrics_manager.get_metrics(),
            }
            logger.debug(f"Retrieved Supervisor status: {status}")
            return status
        except Exception as e:
            logger.error(f"Error getting Supervisor status: {e}")
            return None

    def get_config_class(self, provider_type):
        """Get the configuration class for a specific LLM provider type.

        Returns the appropriate configuration class based on the provider type.
        This method provides a mapping from provider names to their corresponding
        configuration classes.

        Args:
            provider_type: The type of LLM provider (e.g., 'openai', 'anthropic', 'bedrock').

        Returns:
            The configuration class for the specified provider, or None if unsupported.

        Note:
            This method is marked for refactoring to use dynamic type extraction
            instead of hardcoded mappings.
        """
        # TODO: [Low] We need to get rid of this method and extract the type dynamically, we don't want to be tied to hard coded definitions

        if provider_type == "openai":
            return OpenAIConfig
        elif provider_type == "anthropic":
            return AnthropicConfig
        elif provider_type == "bedrock":
            return BedrockConfig
        else:
            return None

    async def _create_heartbeat_task(self):
        """Create scheduled heartbeat task for system health monitoring."""
        while True:
            try:
                # Publish system heartbeat event
                if self.message_bus and self.message_bus.is_running():
                    self.message_bus.publish(
                        publisher=self,
                        message_type="HEARTBEAT_TICK",
                        message={"timestamp": time.time()},
                    )
                    logger.debug("System heartbeat tick published (30s interval)")

                # Wait for next heartbeat (30-second intervals)
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Heartbeat task error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _create_fast_heartbeat_task(self):
        """Create scheduled fast heartbeat task."""
        tick_count = 0
        while True:
            try:
                # Publish fast heartbeat tick directly through message bus
                if self.message_bus and self.message_bus.is_running():
                    self.message_bus.publish(
                        publisher=self,
                        message_type="FAST_HEARTBEAT_TICK",
                        message={"tick": tick_count, "timestamp": time.time()},
                    )
                    tick_count += 1
                    logger.debug(
                        f"Fast heartbeat tick {tick_count} published (5s interval)"
                    )

                # Wait for next tick (5 seconds for timer monitoring)
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Fast heartbeat task error: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

    def _start_scheduled_tasks(self):
        """Start scheduled tasks for heartbeat operations."""
        try:
            loop = asyncio.get_running_loop()
            # Create and schedule the heartbeat tasks
            asyncio.create_task(self._create_heartbeat_task())
            asyncio.create_task(self._create_fast_heartbeat_task())
            logger.info("Heartbeat tasks started successfully in event loop")
        except RuntimeError:
            # No event loop available yet - will be started when async context is available
            logger.warning(
                "No event loop available - heartbeat tasks will start with async context"
            )

    def _stop_scheduled_tasks(self):
        """Stop scheduled tasks."""
        if self._scheduled_tasks:
            logger.info(f"Clearing {len(self._scheduled_tasks)} scheduled tasks")
            self._scheduled_tasks.clear()

    def _setup_message_bus_dependencies(self):
        """Set up MessageBus dependencies for intent processing."""
        if self.message_bus and hasattr(self.message_bus, "set_dependencies"):
            self.message_bus.set_dependencies(
                communication_manager=self.communication_manager,
                workflow_engine=self.workflow_engine,
                llm_factory=self.llm_factory,
            )
            logger.info("MessageBus dependencies configured for intent processing")

    def _ensure_event_loop(self):
        """Ensure we have an event loop for scheduled tasks."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            # No running loop, try to get the event loop for this thread
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    # Create a new event loop if the current one is closed
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop
            except RuntimeError:
                # Create a new event loop as last resort
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop


if __name__ == "__main__":
    logger.info("Starting Supervisor application...")
    config_file = "config.yaml"
    supervisor = Supervisor(config_file)
    supervisor.run()
    logger.info("Supervisor application stopped.")
