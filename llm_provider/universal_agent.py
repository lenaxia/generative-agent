import logging
import time
from typing import Any, Dict, List, Optional

# Import StrandsAgent - hard dependency, no fallbacks
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands_tools import calculator, file_read, shell

from common.task_context import TaskContext
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.mcp_client import MCPClientManager
from llm_provider.role_registry import RoleDefinition, RoleRegistry
from llm_provider.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class UniversalAgent:
    """
    Universal Agent that can assume different roles using StrandsAgent framework.

    This class provides a unified interface for creating role-specific agents
    while leveraging the semantic model types and prompt library from StrandsAgent.
    """

    def __init__(
        self,
        llm_factory: LLMFactory,
        role_registry: Optional[RoleRegistry] = None,
        mcp_manager: Optional[MCPClientManager] = None,
    ):
        """
        Initialize Universal Agent with LLMFactory, role registry, and optional MCP manager.

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

    def assume_role(
        self,
        role: str,
        llm_type: Optional[LLMType] = None,
        context: Optional[TaskContext] = None,
        tools: Optional[List[str]] = None,
    ):
        """
        Assume role using pooled Agent with context switching.

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

        # Get pooled Agent from LLMFactory (< 0.01s for cache hit)
        llm_type = llm_type or self._determine_llm_type_for_role(role)
        agent = self.llm_factory.get_agent(llm_type)

        # Update Agent context (business logic)
        system_prompt = self._get_system_prompt_from_role(role_def)
        role_tools = self._assemble_role_tools(role_def, tools or [])

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
        """
        Create a StrandsAgent model based on LLM type.

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
        """
        Get role-specific system prompt.

        Args:
            role: Agent role

        Returns:
            str: System prompt for the role
        """
        role_prompts = {
            "planning": """You are a planning specialist agent. Your role is to:
1. Break down complex tasks into manageable steps
2. Create detailed task plans with dependencies
3. Identify required resources and constraints
4. Provide structured planning output
Focus on creating clear, actionable plans.""",
            "search": """You are a search specialist agent. Your role is to:
1. Perform web searches for information
2. Find relevant and accurate information
3. Summarize search results clearly
4. Provide source citations
Focus on finding the most relevant and up-to-date information.""",
            "weather": """You are a weather information specialist agent. Your role is to:
1. Retrieve current weather conditions
2. Provide weather forecasts
3. Explain weather patterns and phenomena
4. Give location-specific weather data
Focus on providing accurate, current weather information.""",
            "summarizer": """You are a text summarization specialist agent. Your role is to:
1. Create concise summaries of long texts
2. Extract key points and main ideas
3. Maintain important context and details
4. Provide structured summary output
Focus on creating clear, comprehensive summaries.""",
            "slack": """You are a Slack integration specialist agent. Your role is to:
1. Send messages to Slack channels
2. Format messages appropriately for Slack
3. Handle Slack-specific formatting and mentions
4. Manage Slack workspace interactions
Focus on effective Slack communication.""",
            "coding": """You are a coding specialist agent. Your role is to:
1. Write clean, efficient code
2. Debug and fix code issues
3. Explain code functionality
4. Follow best practices and patterns
Focus on producing high-quality, maintainable code.""",
            "analysis": """You are an analysis specialist agent. Your role is to:
1. Analyze data and information thoroughly
2. Identify patterns and insights
3. Provide detailed analytical reports
4. Make data-driven recommendations
Focus on comprehensive, accurate analysis.""",
        }

        return role_prompts.get(
            role,
            "You are a helpful AI assistant. Provide accurate, helpful responses to user queries.",
        )

    def execute_task(
        self,
        instruction: str,
        role: str = "default",
        llm_type: LLMType = LLMType.DEFAULT,
        context: Optional[TaskContext] = None,
        extracted_parameters: Optional[Dict] = None,
    ) -> str:
        """
        Enhanced task execution with hybrid role lifecycle support.

        Args:
            instruction: Task instruction
            role: Agent role to assume
            llm_type: Model type for optimization
            context: Optional task context
            extracted_parameters: Parameters extracted during routing

        Returns:
            str: Task result
        """
        # Check execution type
        execution_type = self.role_registry.get_role_execution_type(role)

        if execution_type == "hybrid":
            # Run async hybrid execution in sync context
            import asyncio

            try:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, create a new task
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self._execute_hybrid_task(
                                instruction, role, context, extracted_parameters
                            ),
                        )
                        return future.result()
                else:
                    # If no loop is running, use asyncio.run
                    return asyncio.run(
                        self._execute_hybrid_task(
                            instruction, role, context, extracted_parameters
                        )
                    )
            except RuntimeError:
                # Fallback: run in new event loop
                return asyncio.run(
                    self._execute_hybrid_task(
                        instruction, role, context, extracted_parameters
                    )
                )
        elif execution_type == "programmatic":
            return self.execute_programmatic_task(instruction, role, context)
        else:
            return self.execute_llm_task(instruction, role, llm_type, context)

    def is_programmatic_role(self, role: str) -> bool:
        """
        Check if role should use programmatic execution.

        Args:
            role: Role name to check

        Returns:
            bool: True if role uses programmatic execution
        """
        return self.role_registry.is_programmatic_role(role)

    def get_role_type(self, role: str) -> str:
        """
        Get the execution type for a role.

        Args:
            role: Role name

        Returns:
            str: "programmatic" or "llm"
        """
        return self.role_registry.get_role_type(role)

    def execute_programmatic_task(
        self, instruction: str, role: str, context: Optional[TaskContext] = None
    ) -> str:
        """
        Execute task using programmatic role (no LLM processing).

        Args:
            instruction: Task instruction
            role: Programmatic role name
            context: Optional task context

        Returns:
            str: Serialized task result
        """
        try:
            programmatic_role = self.role_registry.get_programmatic_role(role)
            if not programmatic_role:
                return f"Programmatic role '{role}' not found"

            result = programmatic_role.execute(instruction, context)
            return self._serialize_result(result)

        except Exception as e:
            logger.error(f"Programmatic execution failed for role '{role}': {e}")
            return f"Programmatic execution error: {str(e)}"

    def execute_llm_task(
        self,
        instruction: str,
        role: str,
        llm_type: LLMType,
        context: Optional[TaskContext] = None,
    ) -> str:
        """
        Execute task using LLM-based role (current implementation).

        Args:
            instruction: Task instruction
            role: LLM role name
            llm_type: Model type for optimization
            context: Optional task context

        Returns:
            str: Task result
        """
        # Assume the specified role
        agent = self.assume_role(role, llm_type, context)

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

    def get_status(self) -> Dict[str, Any]:
        """
        Get current Universal Agent status.

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

    def get_available_roles(self) -> List[str]:
        """
        Get list of available agent roles.

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

    def get_mcp_status(self) -> Dict[str, Any]:
        """
        Get MCP integration status for heartbeat monitoring.

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
        """
        Create model with role-specific configuration merged with model capabilities.
        Uses LLM Factory caching for performance.

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
        self, role_config: Dict, base_model: BedrockModel
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

    def _get_system_prompt_from_role(self, role_def: RoleDefinition) -> str:
        """Get system prompt from role definition."""
        prompts = role_def.config.get("prompts", {})
        return prompts.get("system", "You are a helpful AI assistant.")

    def _assemble_role_tools(
        self, role_def: RoleDefinition, additional_tools: List[str]
    ) -> List:
        """
        Assemble all tools for a role from multiple sources.

        Args:
            role_def: Role definition
            additional_tools: Additional tool names to include

        Returns:
            List of tool functions
        """
        tools = []

        # 1. Add built-in StrandsAgent tools
        builtin_tools = [calculator, file_read, shell]
        tools.extend(builtin_tools)

        # 2. Auto-include ALL custom tools from role's tools.py
        tools.extend(role_def.custom_tools)

        # 3. Add specified shared tools
        shared_tool_names = role_def.config.get("tools", {}).get("shared", [])
        for tool_name in shared_tool_names:
            shared_tool = self.role_registry.get_shared_tool(tool_name)
            if shared_tool:
                tools.append(shared_tool)
            else:
                logger.debug(
                    f"Shared tool '{tool_name}' not found for role '{role_def.name}' - may be available as custom tool"
                )

        # 4. Add additional tools from registry
        additional_role_tools = self.tool_registry.get_tools(additional_tools)
        tools.extend(additional_role_tools)

        # 5. Add MCP tools if available
        if self.mcp_manager:
            mcp_tools = self.mcp_manager.get_tools_for_role(role_def.name)
            tools.extend(mcp_tools)

        # 6. TODO: Add automatically selected tools if enabled
        if role_def.config.get("tools", {}).get("automatic", False):
            # This would use LLM to select additional tools
            # For now, we'll implement this in a future iteration
            logger.info(
                f"Automatic tool selection enabled for role '{role_def.name}' (not yet implemented)"
            )

        return tools

    def _create_basic_agent(
        self,
        llm_type: LLMType,
        context: Optional[TaskContext] = None,
        tools: Optional[List[str]] = None,
    ) -> Agent:
        """
        Create a basic agent with default configuration.

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

        agent = Agent(model=model, system_prompt=system_prompt, tools=basic_tools)

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

    def register_programmatic_role(self, name: str, role_instance: "ProgrammaticRole"):
        """
        Register a programmatic role for direct execution.

        Args:
            name: Role name
            role_instance: ProgrammaticRole instance
        """
        self.role_registry.register_programmatic_role(role_instance)

    def _update_agent_context(self, agent: Agent, system_prompt: str, tools: List[Any]):
        """
        Update Agent context with optimal reuse strategy.

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
                new_agent = Agent(model=model, system_prompt=system_prompt, tools=tools)
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

                logger.debug(f"✅ Updated agent context in place (same tools)")
                return agent  # Return same agent instance

        except Exception as e:
            logger.warning(
                f"Failed to update agent context: {e}, recreating with cached model"
            )
            # Fallback: recreate Agent with cached model
            model = agent.model if hasattr(agent, "model") else None
            return Agent(model=model, system_prompt=system_prompt, tools=tools)

    def _determine_llm_type_for_role(self, role: str) -> LLMType:
        """
        Determine the appropriate LLM type for a given role.

        Args:
            role: Role name

        Returns:
            LLMType: Appropriate model type for the role
        """
        # Role-specific LLM type mapping for performance optimization
        role_llm_mapping = {
            # Fast routing roles use WEAK models
            "router": LLMType.WEAK,
            # Complex planning roles use STRONG models
            "planning": LLMType.STRONG,
            "analysis": LLMType.STRONG,
            "coding": LLMType.STRONG,
            # Standard roles use DEFAULT models
            "weather": LLMType.DEFAULT,
            "timer": LLMType.DEFAULT,
            "calendar": LLMType.DEFAULT,
            "search": LLMType.DEFAULT,
            "summarizer": LLMType.DEFAULT,
            "slack": LLMType.DEFAULT,
            "default": LLMType.DEFAULT,
        }

        return role_llm_mapping.get(role, LLMType.DEFAULT)
        logger.info(f"Registered programmatic role: {name}")

    def _serialize_result(self, result: Any) -> str:
        """
        Serialize programmatic results to string format.

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
        extracted_parameters: Optional[Dict],
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
                llm_result = self._execute_llm_with_context(
                    enhanced_instruction, role, context
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
        lifecycle_functions: Dict,
        instruction: str,
        context: TaskContext,
        parameters: Dict,
    ) -> Dict[str, Any]:
        """Run all pre-processing functions for a role."""
        results = {}

        pre_config = role_def.config.get("lifecycle", {}).get("pre_processing", {})
        functions = pre_config.get("functions", [])

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
        lifecycle_functions: Dict,
        llm_result: str,
        context: TaskContext,
        pre_data: Dict,
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
        self, role_def: RoleDefinition, instruction: str, pre_data: Dict
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

    def _flatten_pre_data(self, pre_data: Dict) -> Dict[str, Any]:
        """Flatten pre-processing data for prompt formatting."""
        flattened = {}
        for func_name, data in pre_data.items():
            if isinstance(data, dict) and "error" not in data:
                # Flatten successful results
                for key, value in data.items():
                    flattened[key] = value
        return flattened

    def _format_pre_data_result(self, pre_data: Dict) -> str:
        """Format pre-processing data as final result (for programmatic-only execution)."""
        return str(pre_data)

    def _execute_llm_with_context(
        self, instruction: str, role: str, context: Optional[TaskContext]
    ) -> str:
        """Execute LLM with enhanced context."""
        return self.execute_llm_task(instruction, role, LLMType.DEFAULT, context)
