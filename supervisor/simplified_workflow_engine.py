"""
Simplified Workflow Engine for Phase 4 Meta-Planning.

This module provides a streamlined workflow engine that replaces the complex
DAG-based execution with a single autonomous agent approach. Instead of managing
task graphs and dependencies, it uses meta-planning to create custom agents
that execute independently with selected tools.

Key simplifications:
- No DAG execution or task scheduling
- No checkpointing or result sharing
- No progressive summarization
- Single agent execution model
- Intent-based side effects (same as before)

Execution Flow:
1. Meta-planning: Analyze request and design agent configuration
2. Agent creation: Create runtime agent with selected tools
3. Agent execution: Run agent autonomously (10-15 iterations)
4. Intent processing: Collect and process intents for side effects
5. Result return: Return agent's final output
"""

import logging
from dataclasses import dataclass
from typing import Any

from common.intent_collector import clear_current_collector, get_current_collector
from common.intent_processor import IntentProcessor
from common.task_context import TaskContext
from llm_provider.runtime_agent_factory import RuntimeAgentFactory

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """Result from workflow execution.

    Attributes:
        response: Final text response from agent execution
        metadata: Additional metadata about execution (e.g., iteration count, tool usage)
        success: Whether the workflow completed successfully
        error: Error message if workflow failed
    """

    response: str
    metadata: dict[str, Any] | None = None
    success: bool = True
    error: str | None = None


class SimplifiedWorkflowEngine:
    """
    Simplified workflow engine for Phase 4 meta-planning.

    This engine replaces the complex DAG-based workflow execution with a
    streamlined approach using dynamically configured agents. Instead of
    managing task graphs, it delegates execution to a single autonomous agent
    created at runtime with exactly the tools needed for the request.

    Architecture:
    - Planning Role: Analyzes request and creates AgentConfiguration
    - RuntimeAgentFactory: Creates custom Strands Agent with selected tools
    - Autonomous Agent: Executes independently using tools (max 10-15 iterations)
    - Intent System: Collects and processes intents for side effects

    Benefits:
    - Simpler: No DAG complexity, checkpointing, or result sharing
    - Flexible: Any tool combination via runtime selection
    - Observable: Intent system preserved for monitoring
    - Maintainable: Single execution path, easier debugging
    """

    def __init__(
        self,
        agent_factory: RuntimeAgentFactory,
        intent_processor: IntentProcessor,
    ):
        """Initialize the simplified workflow engine.

        Args:
            agent_factory: Factory for creating runtime agents from configurations
            intent_processor: Processor for handling collected intents
        """
        self.agent_factory = agent_factory
        self.intent_processor = intent_processor

        logger.info("SimplifiedWorkflowEngine initialized")

    async def execute_complex_request(
        self,
        request: str,
        agent_config: Any,  # AgentConfiguration from meta-planning
        context: TaskContext | None = None,
    ) -> WorkflowResult:
        """
        Execute complex request using dynamically configured agent.

        This is the main entry point for Phase 4 workflow execution.
        It follows a simplified 4-step process:

        1. Create runtime agent from configuration
        2. Run agent autonomously with selected tools
        3. Collect intents generated during execution
        4. Process intents and return result

        Args:
            request: User's original request text
            agent_config: AgentConfiguration from meta-planning role
            context: Optional task context for state management

        Returns:
            WorkflowResult with agent's response and execution metadata

        Raises:
            Exception: If agent creation or execution fails
        """
        logger.info(f"Starting simplified workflow for request: {request[:100]}...")

        try:
            # Step 1: Create runtime agent from configuration
            logger.info(
                f"Creating runtime agent with {len(agent_config.tool_names)} tools"
            )
            runtime_agent, intent_collector = self.agent_factory.create_agent(
                config=agent_config,
                context=context,
            )

            # Step 2: Run agent autonomously
            logger.info(
                f"Running agent with max {agent_config.max_iterations} iterations"
            )
            result = await runtime_agent.run(request)

            # Log execution metadata
            logger.info(
                f"Agent execution completed. "
                f"Output length: {len(result.final_output)} chars"
            )

            # Step 3: Collect intents from execution
            intents = intent_collector.get_intents()
            logger.info(f"Collected {len(intents)} intents from execution")

            # Step 4: Process intents (side effects)
            if intents:
                logger.info("Processing collected intents...")
                await self.intent_processor.process_intents(intents)
                logger.info("Intents processed successfully")

            # Clean up intent collector
            clear_current_collector()

            # Return successful result
            return WorkflowResult(
                response=result.final_output,
                metadata={
                    "tool_count": len(agent_config.tool_names),
                    "tools_used": agent_config.tool_names,
                    "max_iterations": agent_config.max_iterations,
                    "intent_count": len(intents),
                },
                success=True,
            )

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)

            # Clean up on error
            clear_current_collector()

            # Return error result
            return WorkflowResult(
                response=f"I encountered an error while processing your request: {str(e)}",
                metadata={
                    "error_type": type(e).__name__,
                    "agent_config": agent_config.to_dict()
                    if hasattr(agent_config, "to_dict")
                    else str(agent_config),
                },
                success=False,
                error=str(e),
            )

    async def execute_simple_request(
        self,
        request: str,
        role_name: str,
        context: TaskContext | None = None,
    ) -> WorkflowResult:
        """
        Execute simple request using predefined role (fast path).

        This method is for high-confidence requests that map directly to
        a single predefined role (weather, timer, calendar, etc.). It bypasses
        meta-planning and creates a minimal agent configuration.

        Args:
            request: User's request text
            role_name: Name of the predefined role to use
            context: Optional task context

        Returns:
            WorkflowResult with role's response

        Note:
            This is a compatibility method during Phase 4 migration.
            Eventually, fast-path roles may execute differently.
        """
        logger.info(f"Executing fast-path request with role: {role_name}")

        # For now, this is a placeholder
        # The actual fast-path execution still uses the existing role system
        # This method will be implemented fully when integrating with supervisor

        return WorkflowResult(
            response=f"Fast-path execution for {role_name} (not yet implemented)",
            metadata={"role": role_name, "fast_path": True},
            success=False,
            error="Fast-path not yet integrated",
        )

    def get_status(self) -> dict[str, Any]:
        """
        Get current engine status and statistics.

        Returns:
            Dictionary with engine status information
        """
        return {
            "engine_type": "SimplifiedWorkflowEngine",
            "phase": 4,
            "features": {
                "meta_planning": True,
                "dynamic_agents": True,
                "dag_execution": False,
                "checkpointing": False,
                "result_sharing": False,
            },
        }
