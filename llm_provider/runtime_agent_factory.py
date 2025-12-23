"""Runtime Agent Factory Module

Creates dynamically configured agents at runtime from AgentConfiguration.
Bridges meta-planning output to executable Strands Agent instances.
"""

import logging
from typing import Any, Tuple

from strands import Agent

from common.agent_configuration import AgentConfiguration
from common.intent_collector import IntentCollector, set_current_collector
from common.task_context import TaskContext
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class RuntimeAgentFactory:
    """Factory for creating runtime agents from configurations.

    This factory takes AgentConfiguration output from meta-planning
    and creates executable Strands Agent instances with:
    - Selected tools from ToolRegistry
    - Enhanced system prompt combining plan + guidance
    - Context-local IntentCollector for execution
    """

    def __init__(self, tool_registry: ToolRegistry, llm_factory: LLMFactory):
        """Initialize the runtime agent factory.

        Args:
            tool_registry: Central tool registry for loading tools
            llm_factory: LLM factory for creating models
        """
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory

        logger.debug("RuntimeAgentFactory initialized")

    def create_agent(
        self,
        config: AgentConfiguration,
        context: TaskContext | None = None,
        llm_type: LLMType = LLMType.DEFAULT,
    ) -> Tuple[Agent, IntentCollector]:
        """Create runtime agent from configuration.

        Args:
            config: Agent configuration from meta-planning
            context: Optional task context for state management
            llm_type: LLM type for model selection

        Returns:
            Tuple of (Strands Agent, IntentCollector) for execution
        """
        logger.info(
            f"Creating runtime agent with {len(config.tool_names)} tools, "
            f"max_iterations={config.max_iterations}"
        )

        # Validate configuration
        if not config.validate():
            raise ValueError("Invalid agent configuration provided")

        # 1. Load tools from registry
        tools = self._load_tools(config.tool_names)
        logger.info(f"Loaded {len(tools)} tools from registry")

        # 2. Build enhanced system prompt
        system_prompt = self._build_system_prompt(config, context)
        logger.debug(f"Built system prompt (length: {len(system_prompt)} chars)")

        # 3. Create model using LLM factory
        model = self.llm_factory.create_strands_model(llm_type)

        # 4. Create intent collector for this execution
        intent_collector = IntentCollector()
        set_current_collector(intent_collector)
        logger.debug("Intent collector created and set for current context")

        # 5. Create Strands Agent
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            # Note: hooks could be added here for intent processing
            # For now, tools will use register_intent() directly
        )

        logger.info(
            f"Runtime agent created successfully with {len(tools)} tools, "
            f"max_iterations={config.max_iterations}"
        )

        return agent, intent_collector

    def _load_tools(self, tool_names: list[str]) -> list[Any]:
        """Load tools from registry by fully qualified names.

        Args:
            tool_names: List of fully qualified tool names (e.g., "weather.get_forecast")

        Returns:
            List of tool objects from registry
        """
        if not tool_names:
            logger.warning("No tools specified in configuration")
            return []

        tools = []
        for tool_name in tool_names:
            tool = self.tool_registry.get_tool(tool_name)
            if tool is not None:
                tools.append(tool)
                logger.debug(f"Loaded tool: {tool_name}")
            else:
                logger.warning(
                    f"Tool '{tool_name}' not found in registry, skipping"
                )

        if len(tools) < len(tool_names):
            missing = len(tool_names) - len(tools)
            logger.warning(
                f"{missing} tools could not be loaded from registry"
            )

        return tools

    def _build_system_prompt(
        self, config: AgentConfiguration, context: TaskContext | None = None
    ) -> str:
        """Build enhanced system prompt from configuration and context.

        Combines:
        - Agent role definition (from config.system_prompt)
        - Step-by-step plan (from config.plan)
        - Specific guidance/constraints (from config.guidance)
        - Optional context information

        Args:
            config: Agent configuration with prompts and plans
            context: Optional task context for additional information

        Returns:
            Enhanced system prompt for agent
        """
        prompt_parts = []

        # 1. Base system prompt (agent role and behavior)
        if config.system_prompt:
            prompt_parts.append(config.system_prompt)

        # 2. Execution plan
        if config.plan:
            prompt_parts.append("\n## EXECUTION PLAN\n")
            prompt_parts.append(
                "Follow this step-by-step plan to complete the task:\n"
            )
            prompt_parts.append(config.plan)

        # 3. Specific guidance/constraints
        if config.guidance:
            prompt_parts.append("\n## GUIDANCE & CONSTRAINTS\n")
            prompt_parts.append(config.guidance)

        # 4. Context information (if available)
        if context:
            context_info = self._extract_context_info(context)
            if context_info:
                prompt_parts.append("\n## CONTEXT\n")
                prompt_parts.append(context_info)

        # 5. Iteration limit reminder
        prompt_parts.append(f"\n## EXECUTION LIMITS\n")
        prompt_parts.append(
            f"You may use tools up to {config.max_iterations} times. "
            f"Plan your tool usage efficiently.\n"
        )

        return "\n".join(prompt_parts)

    def _extract_context_info(self, context: TaskContext) -> str:
        """Extract relevant information from task context.

        Args:
            context: Task context with state and metadata

        Returns:
            Formatted context information string
        """
        info_parts = []

        # User information
        if hasattr(context, "user_id") and context.user_id:
            info_parts.append(f"User: {context.user_id}")

        # Channel/source information
        if hasattr(context, "channel_id") and context.channel_id:
            info_parts.append(f"Channel: {context.channel_id}")

        # Original request (if available)
        if hasattr(context, "original_prompt") and context.original_prompt:
            info_parts.append(f"Original Request: {context.original_prompt}")

        # Metadata (if available)
        if hasattr(context, "metadata") and context.metadata:
            metadata_str = ", ".join(
                f"{k}={v}" for k, v in context.metadata.items()
            )
            info_parts.append(f"Metadata: {metadata_str}")

        return "\n".join(info_parts) if info_parts else ""
