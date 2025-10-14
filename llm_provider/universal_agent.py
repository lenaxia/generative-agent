"""Universal Agent module for unified LLM interaction and role execution.

This module provides the UniversalAgent class that serves as the main interface
for executing roles, managing hybrid execution modes, and coordinating between
different LLM providers and execution strategies.
"""

import logging
import time
from enum import Enum
from typing import Any, Optional

from strands import Agent
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import AfterToolCallEvent


class ExecutionMode(str, Enum):
    """Execution modes for role-based tool selection."""

    FAST_REPLY = "fast_reply"
    WORKFLOW = "workflow"


from strands.models.bedrock import BedrockModel
from strands_tools import calculator, file_read, shell

from common.enhanced_event_context import LLMSafeEventContext
from common.task_context import TaskContext
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.mcp_client import MCPClientManager
from llm_provider.role_registry import RoleDefinition, RoleRegistry
from llm_provider.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class IntentProcessingHook(HookProvider):
    """Hook provider to process intents from tool results."""

    def __init__(self, universal_agent):
        self.universal_agent = universal_agent
        self.current_context = None

    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(AfterToolCallEvent, self._process_tool_result_intents)

    def _process_tool_result_intents(self, event):
        """Process intents from tool results."""
        try:
            tool_result = event.result

            # Check if tool result contains an intent
            if isinstance(tool_result, dict) and "intent" in tool_result:
                intent_data = tool_result["intent"]

                # Inject context into intent if available
                if self.current_context:
                    intent_data["user_id"] = self.current_context.user_id
                    intent_data["channel_id"] = self.current_context.channel_id

                # Create intent object and process it
                intent = self._create_intent_from_data(intent_data)
                if intent:
                    # Process intent asynchronously
                    import asyncio

                    asyncio.create_task(self._process_intent_async(intent))

        except Exception as e:
            logger.error(f"Error processing tool result intent: {e}")

    def _create_intent_from_data(self, intent_data: dict):
        """Create intent object from tool result data."""
        intent_type = intent_data.get("type")

        if intent_type == "TimerCreationIntent":
            from roles.timer_single_file import TimerCreationIntent

            return TimerCreationIntent(
                **{k: v for k, v in intent_data.items() if k != "type"}
            )
        elif intent_type == "TimerCancellationIntent":
            from roles.timer_single_file import TimerCancellationIntent

            return TimerCancellationIntent(
                **{k: v for k, v in intent_data.items() if k != "type"}
            )
        elif intent_type == "TimerListingIntent":
            from roles.timer_single_file import TimerListingIntent

            return TimerListingIntent(
                **{k: v for k, v in intent_data.items() if k != "type"}
            )

        return None

    async def _process_intent_async(self, intent):
        """Process intent using the registered intent handlers."""
        try:
            # Get intent processor from role registry
            if (
                hasattr(self.universal_agent, "role_registry")
                and self.universal_agent.role_registry
            ):
                role_registry = self.universal_agent.role_registry

                # Find the intent handler for this intent type
                for role_name, role_def in role_registry.roles.items():
                    if hasattr(role_def, "_intent_handlers"):
                        intent_handlers = role_def._intent_handlers
                        if type(intent) in intent_handlers:
                            handler = intent_handlers[type(intent)]
                            await handler(intent)
                            logger.info(
                                f"Processed {type(intent).__name__} via {role_name} role"
                            )
                            return

                logger.warning(
                    f"No handler found for intent type: {type(intent).__name__}"
                )

        except Exception as e:
            logger.error(f"Error processing intent: {e}")


class UniversalAgent:
    r"""This class provides a unified interface for creating role-specific agents
    while leveraging the semantic model types and prompt library from StrandsAgent.
    """

    def __init__(
        self,
        llm_factory: LLMFactory,
        role_registry: Optional[RoleRegistry] = None,
        mcp_manager: Optional[MCPClientManager] = None,
    ):
        r"""\1

        Args:
            llm_factory: Enhanced LLMFactory instance
            role_registry: Role registry for dynamic role loading
            mcp_manager: Optional MCP client manager for external tools
        """
        self.llm_factory = llm_factory
        self.tool_registry = ToolRegistry()
        self.role_registry = role_registry or RoleRegistry.get_global_registry()
        self.mcp_manager = mcp_manager
        self.current_agent = None
        self.current_role = None
        self.current_llm_type = None

        # Initialize intent processing hook
        self.intent_hook = IntentProcessingHook(self)

    def assume_role(
        self,
        role: str,
        llm_type: Optional[LLMType] = None,
        context: Optional[TaskContext] = None,
        tools: Optional[list[str]] = None,
        execution_mode: Optional[ExecutionMode] = None,
    ):
        r"""\1

        Args:
            role: The role name (e.g., 'weather', 'timer', 'calendar')
            llm_type: Semantic model type for performance optimization
            context: Optional TaskContext for state management
            tools: Optional list of additional tool names

        Returns:
            Pooled Agent instance with updated context
        """
        # Handle role fallbacks
        if role == "None" or role is None:
            logger.info("None role requested, falling back to default role")
            role = "default"

        # Load role definition (cached by RoleRegistry)
        role_def = self.role_registry.get_role(role)
        if not role_def:
            logger.warning(f"Role '{role}' not found, falling back to default")
            role = "default"
            role_def = self.role_registry.get_role(role)

        # If even default role doesn't exist, create a basic fallback
        if not role_def:
            logger.warning("Default role not found, creating basic fallback agent")
            llm_type = llm_type or LLMType.DEFAULT
            agent = self.llm_factory.get_agent(llm_type)

            # Create basic agent with minimal configuration
            basic_agent = self._update_agent_context(
                agent, "You are a helpful AI assistant.", []
            )

            # Store current configuration
            self.current_agent = basic_agent if basic_agent is not None else agent
            self.current_role = "basic"  # Mark as basic fallback role
            self.current_llm_type = llm_type

            return self.current_agent

        # Get pooled Agent from LLMFactory (< 0.01s for cache hit)
        llm_type = llm_type or self._determine_llm_type_for_role(role)
        agent = self.llm_factory.get_agent(llm_type)

        # Update Agent context (business logic)
        system_prompt = self._get_system_prompt_from_role(role_def)
        role_tools = self._assemble_role_tools(role_def, tools or [], execution_mode)

        # Context switching instead of Agent creation (< 0.01s)
        updated_agent = self._update_agent_context(agent, system_prompt, role_tools)

        # Use the updated agent (may be the same or a new one)
        final_agent = updated_agent if updated_agent is not None else agent

        # Store current configuration
        self.current_agent = final_agent
        self.current_role = role
        self.current_llm_type = llm_type

        logger.info(f"⚡ Switched to role '{role}' with {len(role_tools)} tools")
        return final_agent

    def _create_strands_model(self, llm_type: LLMType):
        r"""\1

        Args:
            llm_type: Semantic model type

        Returns:
            StrandsAgent model instance
        """
        # Map LLM types to model configurations from config
        model_mapping = {
            LLMType.WEAK: "us.anthropic.claude-sonnet-4-20250514-v1:0",
            LLMType.DEFAULT: "us.anthropic.claude-sonnet-4-20250514-v1:0",
            LLMType.STRONG: "us.anthropic.claude-sonnet-4-20250514-v1:0",
        }

        model_id = model_mapping.get(llm_type, model_mapping[LLMType.DEFAULT])

        # Create Bedrock model with proper configuration
        return BedrockModel(
            model_id=model_id, region_name="us-west-2", temperature=0.3, max_tokens=4096
        )

    def _get_role_prompt(self, role: str) -> str:
        r"""\1

        Args:
            role: Agent role

        Returns:
            str: System prompt for the role from role definition
        """
        # Get system prompt from role definition (dynamic from YAML)
        role_def = self.role_registry.get_role(role)
        if role_def:
            prompts = role_def.config.get("prompts", {})
            system_prompt = prompts.get("system", "")
            if system_prompt:
                return system_prompt

        # Fallback for roles without definitions
        return "You are a helpful AI assistant. Provide accurate, helpful responses to user queries."

    def execute_task(
        self,
        instruction: str,
        role: str = "default",
        llm_type: LLMType = LLMType.DEFAULT,
        context: Optional[TaskContext] = None,
        event_context: Optional[LLMSafeEventContext] = None,
        extracted_parameters: Optional[dict] = None,
    ) -> str:
        r"""\1

        Args:
            instruction: Task instruction
            role: Agent role to assume
            llm_type: Model type for optimization
            context: Optional task context
            extracted_parameters: Parameters extracted during routing

        Returns:
            str: Task result
        """
        # LLM-SAFE: Single event loop architecture (Documents 25 & 26)
        # Always use synchronous execution to eliminate threading complexity

        # Store event context for intent processing
        if event_context:
            self.intent_hook.current_context = event_context

        return self.execute_llm_task(
            instruction, role, llm_type, context, event_context=event_context
        )

    # Programmatic role methods removed - everything is hybrid now

    def execute_llm_task(
        self,
        instruction: str,
        role: str,
        llm_type: LLMType,
        context: Optional[TaskContext] = None,
        execution_mode: ExecutionMode = ExecutionMode.WORKFLOW,
        event_context: Optional[LLMSafeEventContext] = None,
    ) -> str:
        r"""\1

        Args:
            instruction: Task instruction
            role: LLM role name
            llm_type: Model type for optimization
            context: Optional task context
            execution_mode: Execution mode for tool selection

        Returns:
            str: Task result
        """
        # Assume the specified role with execution mode
        agent = self.assume_role(role, llm_type, context, execution_mode=execution_mode)

        # Execute the task
        try:
            response = agent(instruction)
            if response is None:
                return "No response generated"

            # Handle Strands AgentResult object
            if hasattr(response, "message") and hasattr(response.message, "content"):
                # Extract text content from AgentResult
                content = response.message.content
                if isinstance(content, list) and len(content) > 0:
                    # Get the first text block
                    first_content = content[0]
                    if isinstance(first_content, dict) and "text" in first_content:
                        return first_content["text"]
                    elif hasattr(first_content, "text"):
                        return first_content.text
                elif isinstance(content, str):
                    return content

            # Fallback to string conversion
            return str(response)
        except Exception as e:
            logger.error(f"Error executing task with agent: {e}")
            return f"Error executing task: {str(e)}"

    def get_status(self) -> dict[str, Any]:
        r"""\1

        Returns:
            Dict: Status information
        """
        return {
            "universal_agent_enabled": True,
            "has_llm_factory": self.llm_factory is not None,
            "has_universal_agent": True,
            "mcp_integration": {
                "mcp_available": self.mcp_manager is not None,
                "registered_servers": (
                    self.mcp_manager.get_registered_servers()
                    if self.mcp_manager
                    else []
                ),
                "total_tools": (
                    len(self.mcp_manager.get_all_tools()) if self.mcp_manager else 0
                ),
                "server_configs": (
                    self.mcp_manager.get_server_configs() if self.mcp_manager else {}
                ),
            },
            "current_role": self.current_role,
            "current_llm_type": (
                self.current_llm_type.value if self.current_llm_type else None
            ),
            "framework": "strands",
        }

    def get_available_roles(self) -> list[str]:
        r"""\1

        Returns:
            List[str]: Available roles
        """
        return [
            "planning",
            "search",
            "weather",
            "summarizer",
            "slack",
            "coding",
            "analysis",
            "default",
        ]

    def get_mcp_status(self) -> dict[str, Any]:
        r"""\1

        Returns:
            Dict: MCP status information
        """
        if self.mcp_manager:
            return {
                "mcp_available": True,
                "registered_servers": self.mcp_manager.get_registered_servers(),
                "total_tools": len(self.mcp_manager.get_all_tools()),
                "server_configs": self.mcp_manager.get_server_configs(),
            }
        else:
            return {
                "mcp_available": False,
                "registered_servers": [],
                "total_tools": 0,
                "server_configs": {},
            }

    def _create_model_for_role(
        self, role_def: RoleDefinition, llm_type: LLMType
    ) -> BedrockModel:
        r"""Uses LLM Factory caching for performance.

        Args:
            role_def: Role definition containing model config
            llm_type: Semantic model type

        Returns:
            Configured StrandsAgent model
        """
        # Use LLM Factory caching for base model creation
        base_model = self.llm_factory.create_strands_model(llm_type)

        # Get role model config for any customizations
        role_model_config = role_def.config.get("model_config", {})

        # If no role-specific config, return cached model directly
        if not role_model_config:
            return base_model

        # Merge configurations (role preferences with model limits) if needed
        merged_config = self._merge_model_configs(role_model_config, base_model)

        return merged_config

    def _merge_model_configs(
        self, role_config: dict, base_model: BedrockModel
    ) -> BedrockModel:
        """Merge role configuration with base model."""
        try:
            # Handle mocked objects in tests
            if hasattr(base_model, "_mock_name"):
                # Return the base model as-is for mocked objects
                return base_model

            # Extract current model parameters safely
            model_params = {
                "model_id": getattr(
                    base_model, "model_id", "us.anthropic.claude-sonnet-4-20250514-v1:0"
                ),
                "region_name": getattr(base_model, "region_name", "us-west-2"),
                "temperature": role_config.get(
                    "temperature", getattr(base_model, "temperature", 0.3)
                ),
                "max_tokens": min(
                    role_config.get(
                        "max_tokens", getattr(base_model, "max_tokens", 4096)
                    ),
                    getattr(base_model, "max_tokens", 4096),  # Respect model limits
                ),
            }

            # Create new model with merged config
            return BedrockModel(**model_params)

        except Exception as e:
            logger.warning(f"Failed to merge model configs: {e}, using base model")
            return base_model

    def _get_system_prompt_from_role(self, role_def) -> str:
        """Get system prompt from role definition."""
        # Handle both dict and RoleDefinition objects
        if isinstance(role_def, dict):
            prompts = role_def.get("config", {}).get("prompts", {})
        else:
            prompts = role_def.config.get("prompts", {})
        return prompts.get("system", "You are a helpful AI assistant.")

    def _assemble_role_tools(
        self,
        role_def,
        additional_tools: list[str],
        execution_mode: Optional[ExecutionMode] = None,
    ) -> list:
        r"""\1

        Args:
            role_def: Role definition (dict or RoleDefinition object)
            additional_tools: Additional tool names to include
            execution_mode: Execution mode for tool selection

        Returns:
            List of tool functions
        """
        tools = []

        # 1. Add built-in StrandsAgent tools (except for router role)
        role_name = (
            role_def.name
            if hasattr(role_def, "name")
            else role_def.get("name", "unknown")
        )
        # Check role's tools configuration to determine if built-in tools should be included
        if role_def:
            config = (
                role_def.config
                if hasattr(role_def, "config")
                else role_def.get("config", {})
            )
            tools_config = config.get("tools", {})
            include_builtin_tools = tools_config.get(
                "include_builtin", True
            )  # Default to True for backward compatibility

            if include_builtin_tools:
                builtin_tools = [calculator, file_read, shell]
                tools.extend(builtin_tools)
                logger.debug(f"Added built-in tools for role '{role_name}'")
            else:
                logger.debug(
                    f"Excluded built-in tools for role '{role_name}' (role configuration)"
                )
        else:
            # Fallback: include built-in tools if no role definition
            builtin_tools = [calculator, file_read, shell]
            tools.extend(builtin_tools)

        # 2. Handle execution-specific tool configuration
        if role_def:
            if isinstance(role_def, dict):
                custom_tools = role_def.get("custom_tools", [])
                shared_tool_names = (
                    role_def.get("config", {}).get("tools", {}).get("shared", [])
                )
                role_name = role_def.get("name", "unknown")
                config = role_def.get("config", {})
            else:
                custom_tools = getattr(role_def, "custom_tools", [])
                shared_tool_names = role_def.config.get("tools", {}).get("shared", [])
                role_name = role_def.name
                config = role_def.config

            # Execution-specific tool handling
            tools_config = config.get("tools", {})
            should_include_custom_tools = self._should_include_custom_tools(
                tools_config, execution_mode, role_name
            )

            if should_include_custom_tools:
                tools.extend(custom_tools)
                logger.debug(
                    f"Added {len(custom_tools)} custom tools for role '{role_name}' in {execution_mode} mode"
                )
            else:
                logger.debug(
                    f"Skipped {len(custom_tools)} custom tools for role '{role_name}' in {execution_mode} mode"
                )

            # 3. Add specified shared tools
            for tool_name in shared_tool_names:
                shared_tool = self.role_registry.get_shared_tool(tool_name)
                if shared_tool:
                    tools.append(shared_tool)
                else:
                    logger.debug(
                        f"Shared tool '{tool_name}' not found for role '{role_name}' - may be available as custom tool"
                    )

            # 5. Add MCP tools if available
            if self.mcp_manager:
                mcp_tools = self.mcp_manager.get_tools_for_role(role_name)
                tools.extend(mcp_tools)

            # 6. TODO: Add automatically selected tools if enabled
            if config.get("tools", {}).get("automatic", False):
                # This would use LLM to select additional tools
                # For now, we'll implement this in a future iteration
                logger.info(
                    f"Automatic tool selection enabled for role '{role_name}' (not yet implemented)"
                )

        # 4. Add additional tools from registry
        additional_role_tools = self.tool_registry.get_tools(additional_tools)
        tools.extend(additional_role_tools)

        return tools

    def _should_include_custom_tools(
        self,
        tools_config: dict,
        execution_mode: Optional[ExecutionMode],
        role_name: str,
    ) -> bool:
        """Determine if custom tools should be included based on execution mode and configuration.

        Args:
            tools_config: Tools configuration from role definition
            execution_mode: Current execution mode
            role_name: Name of the role for logging

        Returns:
            bool: Whether to include custom tools
        """
        if not execution_mode:
            # No execution mode specified, use automatic setting
            return tools_config.get("automatic", True)

        # Check for execution-specific configuration
        if "execution_modes" in tools_config:
            execution_modes = tools_config["execution_modes"]
            mode_key = execution_mode.value
            if mode_key in execution_modes:
                mode_config = execution_modes[mode_key]
                # If mode_config is a list, it's the available tools
                if isinstance(mode_config, list):
                    return len(mode_config) > 0
                # If mode_config is a dict, check enabled flag
                elif isinstance(mode_config, dict):
                    return mode_config.get("enabled", False)

            # If execution mode not found in config, use default behavior
            logger.debug(
                f"Execution mode '{execution_mode.value}' not configured for role '{role_name}', using default"
            )

        # Default behavior based on execution mode
        if execution_mode == ExecutionMode.FAST_REPLY:
            # Fast-reply modes: no tools by default unless explicitly configured
            fast_reply_tools = tools_config.get("fast_reply", [])
            if isinstance(fast_reply_tools, list):
                return len(fast_reply_tools) > 0
            elif isinstance(fast_reply_tools, dict):
                return fast_reply_tools.get("enabled", False)
            else:
                # No fast_reply configuration, default to no tools for fast-reply
                return False
        else:
            # Workflow mode: use automatic setting (default True for backward compatibility)
            return tools_config.get("automatic", True)

    def _create_basic_agent(
        self,
        llm_type: LLMType,
        context: Optional[TaskContext] = None,
        tools: Optional[list[str]] = None,
    ) -> Agent:
        r"""\1

        Args:
            llm_type: LLM type for the agent
            context: Optional task context
            tools: Optional list of additional tool names

        Returns:
            Agent configured with basic settings
        """
        model = self._create_strands_model(llm_type)

        # Use default system prompt
        system_prompt = """You are a helpful AI assistant. Analyze the task at hand and use the most appropriate tools to complete it effectively. Provide clear, accurate, and helpful responses."""
        # Use basic tools only
        from strands_tools import calculator, file_read, shell

        basic_tools = [calculator, file_read, shell]

        # Add any additional tools from registry if specified
        if tools:
            additional_tools = self.tool_registry.get_tools(tools)
            basic_tools.extend(additional_tools)

        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=basic_tools,
            hooks=[self.intent_hook],
        )

        # Store current configuration
        self.current_agent = agent
        self.current_role = "basic"
        self.current_llm_type = llm_type

        logger.info(f"Created basic agent with {len(basic_tools)} tools")
        return agent

    def reset(self):
        """Reset the Universal Agent state."""
        self.current_agent = None
        self.current_role = None
        self.current_llm_type = None

    # register_programmatic_role method removed - everything is hybrid now

    def _update_agent_context(self, agent: Agent, system_prompt: str, tools: list[Any]):
        r"""\1

        Based on Strands documentation:
        - system_prompt can be updated directly
        - tools require agent recreation
        - conversation history should be cleared for context switches

        Args:
            agent: Pooled Agent instance to update
            system_prompt: New system prompt for the role
            tools: New tool set for the role
        """
        try:
            # Check if we need to recreate due to tool changes
            current_tool_names = getattr(agent, "tool_names", [])
            new_tool_names = [getattr(tool, "__name__", str(tool)) for tool in tools]

            tools_changed = set(current_tool_names) != set(new_tool_names)

            if tools_changed:
                # Tools changed - need to recreate agent but reuse cached model
                model = agent.model
                new_agent = Agent(
                    model=model,
                    system_prompt=system_prompt,
                    tools=tools,
                    hooks=[self.intent_hook],
                )
                logger.debug(
                    f"✅ Recreated agent with cached model due to tool changes: {len(tools)} tools"
                )
                return new_agent
            else:
                # Tools same - just update system prompt and clear history
                agent.system_prompt = system_prompt

                # Clear conversation history for clean context switch (per Strands docs)
                if hasattr(agent, "messages"):
                    agent.messages.clear()
                    logger.debug("✅ Cleared conversation history for context switch")

                # Also clear any conversation manager state if available
                if hasattr(agent, "conversation_manager") and hasattr(
                    agent.conversation_manager, "get_state"
                ):
                    try:
                        # Reset conversation manager state for clean context
                        if hasattr(agent.conversation_manager, "removed_message_count"):
                            agent.conversation_manager.removed_message_count = 0
                        logger.debug("✅ Reset conversation manager state")
                    except Exception as e:
                        logger.debug(f"Could not reset conversation manager state: {e}")

                logger.debug("✅ Updated agent context in place (same tools)")
                return agent  # Return same agent instance

        except Exception as e:
            logger.warning(
                f"Failed to update agent context: {e}, recreating with cached model"
            )
            # Fallback: recreate Agent with cached model
            model = agent.model if hasattr(agent, "model") else None
            return Agent(
                model=model,
                system_prompt=system_prompt,
                tools=tools,
                hooks=[self.intent_hook],
            )

    def _determine_llm_type_for_role(self, role: str) -> LLMType:
        r"""\1

        Args:
            role: Role name

        Returns:
            LLMType: Appropriate model type for the role
        """
        # Get LLM type from role registry (dynamic from YAML definitions)
        llm_type_str = self.role_registry.get_role_llm_type(role)

        # Convert string to LLMType enum
        try:
            return LLMType[llm_type_str.upper()]
        except (KeyError, AttributeError):
            logger.warning(
                f"Invalid LLM type '{llm_type_str}' for role '{role}', using DEFAULT"
            )
            return LLMType.DEFAULT

    def _serialize_result(self, result: Any) -> str:
        r"""\1

        Args:
            result: Result data from programmatic execution

        Returns:
            str: Serialized result
        """
        if isinstance(result, str):
            return result
        elif isinstance(result, (dict, list)):
            import json

            return json.dumps(result, indent=2, default=str)
        else:
            return str(result)

    async def _execute_hybrid_task(
        self,
        instruction: str,
        role: str,
        context: Optional[TaskContext],
        extracted_parameters: Optional[dict],
    ) -> str:
        """Execute hybrid role with lifecycle hooks."""
        import time

        start_time = time.time()

        try:
            role_def = self.role_registry.get_role(role)
            if not role_def:
                raise ValueError(f"Role '{role}' not found")

            lifecycle_functions = self.role_registry.get_lifecycle_functions(role)

            # 1. Pre-processing phase
            pre_data = {}
            if self._has_pre_processing(role_def):
                pre_start_time = time.time()
                logger.info(f"Running pre-processing for {role}")
                pre_data = await self._run_pre_processors(
                    role_def,
                    lifecycle_functions,
                    instruction,
                    context,
                    extracted_parameters or {},
                )
                pre_execution_time = (time.time() - pre_start_time) * 1000
                logger.info(
                    f"Pre-processing for {role} completed in {pre_execution_time:.1f}ms"
                )

            # 2. LLM execution phase (if needed)
            llm_result = None
            if self._needs_llm_processing(role_def):
                logger.info(f"Running LLM processing for {role}")
                enhanced_instruction = self._inject_pre_data(
                    role_def, instruction, pre_data
                )
                # Use WORKFLOW mode for router role to ensure custom tools are available
                execution_mode = (
                    ExecutionMode.WORKFLOW
                    if role == "router"
                    else ExecutionMode.FAST_REPLY
                )
                llm_result = self._execute_llm_with_context(
                    enhanced_instruction,
                    role,
                    context,
                    execution_mode=execution_mode,
                )

            # 3. Post-processing phase
            final_result = llm_result or self._format_pre_data_result(pre_data)
            if self._has_post_processing(role_def):
                post_start_time = time.time()
                logger.info(f"Running post-processing for {role}")
                final_result = await self._run_post_processors(
                    role_def, lifecycle_functions, final_result, context, pre_data
                )
                post_execution_time = (time.time() - post_start_time) * 1000
                logger.info(
                    f"Post-processing for {role} completed in {post_execution_time:.1f}ms"
                )

            execution_time = time.time() - start_time
            logger.info(
                f"Hybrid role {role} completed in {execution_time:.3f}s ({execution_time*1000:.1f}ms)"
            )
            return final_result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Hybrid role {role} failed after {execution_time:.3f}s: {e}")
            return f"Error in {role}: {str(e)}"

    async def _run_pre_processors(
        self,
        role_def: RoleDefinition,
        lifecycle_functions: dict,
        instruction: str,
        context: TaskContext,
        parameters: dict,
    ) -> dict[str, Any]:
        """Run all pre-processing functions for a role."""
        results = {}

        # Handle both multi-file (legacy) and single-file role pre-processors
        pre_config = role_def.config.get("lifecycle", {}).get("pre_processing", {})
        functions = pre_config.get("functions", [])

        # NEW: Handle single-file role pre-processors
        single_file_processors = getattr(role_def, "_pre_processors", [])
        if single_file_processors:
            logger.info(
                f"Found {len(single_file_processors)} single-file pre-processors for {role_def.name}"
            )
            for processor in single_file_processors:
                try:
                    func_name = getattr(processor, "__name__", "unknown_processor")
                    logger.info(f"Running single-file pre-processor: {func_name}")
                    result = processor(instruction, context, parameters)
                    results[func_name] = result
                    logger.info(
                        f"Single-file pre-processor {func_name} completed successfully"
                    )
                except Exception as e:
                    logger.error(f"Single-file pre-processor {func_name} failed: {e}")

        # Handle legacy multi-file pre-processors
        for func_config in functions:
            if isinstance(func_config, str):
                func_name = func_config
                func_params = []
            else:
                func_name = func_config.get("name")
                func_params = func_config.get("uses_parameters", [])

            processor = lifecycle_functions.get(func_name)
            if processor:
                func_start_time = time.time()
                try:
                    # Extract relevant parameters for this function
                    func_parameters = {
                        k: v for k, v in parameters.items() if k in func_params
                    }

                    result = await processor(instruction, context, func_parameters)
                    results[func_name] = result

                    func_execution_time = (time.time() - func_start_time) * 1000
                    logger.debug(
                        f"Pre-processor '{func_name}' completed in {func_execution_time:.1f}ms"
                    )

                except Exception as e:
                    func_execution_time = (time.time() - func_start_time) * 1000
                    logger.error(
                        f"Pre-processor '{func_name}' failed after {func_execution_time:.1f}ms: {e}"
                    )
                    results[func_name] = {"error": str(e)}
            else:
                logger.warning(f"Pre-processor function '{func_name}' not found")

        return results

    async def _run_post_processors(
        self,
        role_def: RoleDefinition,
        lifecycle_functions: dict,
        llm_result: str,
        context: TaskContext,
        pre_data: dict,
    ) -> str:
        """Run all post-processing functions for a role."""
        current_result = llm_result

        post_config = role_def.config.get("lifecycle", {}).get("post_processing", {})
        functions = post_config.get("functions", [])

        for func_config in functions:
            func_name = (
                func_config if isinstance(func_config, str) else func_config.get("name")
            )

            processor = lifecycle_functions.get(func_name)
            if processor:
                func_start_time = time.time()
                try:
                    current_result = await processor(current_result, context, pre_data)

                    func_execution_time = (time.time() - func_start_time) * 1000
                    logger.debug(
                        f"Post-processor '{func_name}' completed in {func_execution_time:.1f}ms"
                    )

                except Exception as e:
                    func_execution_time = (time.time() - func_start_time) * 1000
                    logger.error(
                        f"Post-processor '{func_name}' failed after {func_execution_time:.1f}ms: {e}"
                    )
                    # Continue with current result on post-processor failure
            else:
                logger.warning(f"Post-processor function '{func_name}' not found")

        return current_result

    def _has_pre_processing(self, role_def: RoleDefinition) -> bool:
        """Check if pre-processing is enabled for a role."""
        return (
            role_def.config.get("lifecycle", {})
            .get("pre_processing", {})
            .get("enabled", False)
        )

    def _has_post_processing(self, role_def: RoleDefinition) -> bool:
        """Check if post-processing is enabled for a role."""
        return (
            role_def.config.get("lifecycle", {})
            .get("post_processing", {})
            .get("enabled", False)
        )

    def _needs_llm_processing(self, role_def: RoleDefinition) -> bool:
        """Determine if LLM processing is needed for a role."""
        execution_type = role_def.config.get("role", {}).get("execution_type", "hybrid")
        return execution_type in ["hybrid", "llm"]

    def _inject_pre_data(
        self, role_def: RoleDefinition, instruction: str, pre_data: dict
    ) -> str:
        """Inject pre-processing data into instruction context."""
        system_prompt = role_def.config.get("prompts", {}).get("system", "")

        # Format system prompt with pre-processed data
        try:
            formatted_prompt = system_prompt.format(**self._flatten_pre_data(pre_data))
            return f"{formatted_prompt}\n\nUser Request: {instruction}"
        except KeyError as e:
            logger.warning(f"Failed to format system prompt with pre-data: {e}")
            return instruction

    def _flatten_pre_data(self, pre_data: dict) -> dict[str, Any]:
        """Flatten pre-processing data for prompt formatting."""
        flattened = {}
        for _func_name, data in pre_data.items():
            if isinstance(data, dict) and "error" not in data:
                # Flatten successful results
                for key, value in data.items():
                    flattened[key] = value
        return flattened

    def _format_pre_data_result(self, pre_data: dict) -> str:
        """Format pre-processing data as final result (for programmatic-only execution)."""
        return str(pre_data)

    def _execute_llm_with_context(
        self,
        instruction: str,
        role: str,
        context: Optional[TaskContext],
        execution_mode: ExecutionMode = ExecutionMode.FAST_REPLY,
    ) -> str:
        """Execute LLM with enhanced context and execution mode."""
        return self.execute_llm_task(
            instruction, role, LLMType.DEFAULT, context, execution_mode
        )
