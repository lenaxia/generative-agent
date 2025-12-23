"""Agent Configuration Module

Defines the configuration structure for dynamically created agents.
Output from meta-planning process, input to runtime agent factory.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class AgentConfiguration:
    """
    Configuration for dynamically created agent.

    This dataclass represents the output from the meta-planning process.
    It contains all information needed to create a runtime agent with
    specific tools and execution plan.
    """

    # Natural language step-by-step plan
    plan: str

    # System prompt defining agent's role and behavior
    system_prompt: str

    # Selected tool names (fully qualified, e.g., "weather.get_forecast")
    tool_names: List[str]

    # Specific guidance or constraints for the agent
    guidance: str

    # Maximum tool call iterations allowed
    max_iterations: int

    # Additional metadata from planning process
    metadata: Dict[str, Any]

    def validate(self) -> bool:
        """Validate the agent configuration.

        Returns:
            bool: True if configuration is valid
        """
        # Check required fields are not empty
        if not self.plan or not self.plan.strip():
            return False

        if not self.system_prompt or not self.system_prompt.strip():
            return False

        # Check tool names list
        if not isinstance(self.tool_names, list):
            return False

        # Check max iterations is reasonable
        if self.max_iterations < 1 or self.max_iterations > 50:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "plan": self.plan,
            "system_prompt": self.system_prompt,
            "tool_names": self.tool_names,
            "guidance": self.guidance,
            "max_iterations": self.max_iterations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfiguration":
        """Create AgentConfiguration from dictionary.

        Args:
            data: Dictionary with configuration data

        Returns:
            AgentConfiguration instance
        """
        return cls(
            plan=data.get("plan", ""),
            system_prompt=data.get("system_prompt", ""),
            tool_names=data.get("tool_names", []),
            guidance=data.get("guidance", ""),
            max_iterations=data.get("max_iterations", 10),
            metadata=data.get("metadata", {}),
        )
