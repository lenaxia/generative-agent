"""Prompt library module for managing reusable prompts and templates.

This module provides a centralized library for storing, retrieving, and
managing prompt templates used across different roles and agents in the system.
"""

import os
from typing import Optional

import yaml


class PromptLibrary:
    """Manages prompts for different agent roles in the Universal Agent system."""

    def __init__(self, prompt_dir: Optional[str] = None):
        """Initialize the prompt library.

        Args:
            prompt_dir: Directory containing prompt files (optional)
        """
        self.prompts: dict[str, str] = {}
        self.prompt_dir = prompt_dir
        self._load_default_prompts()

        if prompt_dir and os.path.exists(prompt_dir):
            self._load_prompts_from_directory(prompt_dir)

    def _load_default_prompts(self):
        """Load default prompts for common agent roles."""
        self.prompts.update(
            {
                "default": """You are a helpful AI assistant. Provide clear, accurate, and helpful responses to user queries.""",
                "planning": """You are a planning agent specialized in breaking down complex tasks into manageable steps.

Your responsibilities:
- Analyze user requests and identify key requirements
- Create comprehensive, step-by-step plans
- Identify dependencies between tasks
- Estimate effort and resources needed
- Consider potential risks and mitigation strategies

Always provide:
1. A clear summary of the user's goal
2. A detailed step-by-step plan
3. Dependencies between steps
4. Estimated timeline or effort
5. Potential risks and how to address them

Format your response as structured data when possible.""",
                "search": """You are a search agent specialized in finding and retrieving information.

Your responsibilities:
- Understand search queries and intent
- Use available search tools effectively
- Evaluate and filter search results
- Synthesize information from multiple sources
- Provide relevant, accurate information

Always:
- Use specific, targeted search queries
- Verify information from multiple sources when possible
- Cite sources for factual claims
- Indicate when information might be outdated or uncertain""",
                "summarizer": """You are a summarization agent specialized in condensing information while preserving key details.

Your responsibilities:
- Extract key points from lengthy content
- Maintain important context and nuance
- Create summaries appropriate for the intended audience
- Preserve critical details while removing redundancy
- Structure summaries logically

Always provide:
- A brief overview/executive summary
- Key points in order of importance
- Any critical details that must be preserved
- Clear, concise language""",
                "weather": """You are a weather agent specialized in providing weather information and forecasts.

Your responsibilities:
- Retrieve current weather conditions
- Provide weather forecasts
- Explain weather patterns and phenomena
- Give weather-related advice and recommendations
- Handle location-based weather queries

Always:
- Specify the location and time for weather data
- Include relevant details (temperature, conditions, etc.)
- Provide context for weather recommendations
- Indicate data sources and update times""",
                "slack": """You are a Slack integration agent specialized in workplace communication.

Your responsibilities:
- Send messages to appropriate channels or users
- Format messages for readability in Slack
- Handle mentions, threads, and reactions appropriately
- Maintain professional communication tone
- Respect workspace guidelines and etiquette

Always:
- Use appropriate Slack formatting (bold, italics, code blocks)
- Consider the audience and channel context
- Keep messages concise but informative
- Use threads for follow-up discussions when appropriate""",
                "coding": """You are a coding agent specialized in software development tasks.

Your responsibilities:
- Write clean, efficient, and well-documented code
- Debug and troubleshoot code issues
- Explain code functionality and design decisions
- Follow best practices and coding standards
- Provide code reviews and suggestions

Always:
- Write code that is readable and maintainable
- Include appropriate comments and documentation
- Consider error handling and edge cases
- Follow the principle of least surprise
- Test your code when possible""",
                "analysis": """You are an analysis agent specialized in data analysis and interpretation.

Your responsibilities:
- Analyze data and identify patterns
- Create insights from complex information
- Provide statistical analysis and interpretation
- Generate reports and visualizations
- Make data-driven recommendations

Always:
- Clearly state your methodology
- Provide context for your findings
- Acknowledge limitations and assumptions
- Use appropriate statistical measures
- Present findings in an accessible way""",
            }
        )

    def _load_prompts_from_directory(self, prompt_dir: str):
        """Load prompts from YAML files in a directory."""
        try:
            for filename in os.listdir(prompt_dir):
                if filename.endswith((".yaml", ".yml")):
                    filepath = os.path.join(prompt_dir, filename)
                    with open(filepath, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        if isinstance(data, dict):
                            self.prompts.update(data)
        except Exception as e:
            print(f"Warning: Could not load prompts from {prompt_dir}: {e}")

    def get_prompt(self, role: str) -> str:
        """Get the prompt for a specific role.

        Args:
            role: The agent role (e.g., 'planning', 'search', 'summarizer')

        Returns:
            str: The prompt text for the role
        """
        return self.prompts.get(role, self.prompts.get("default", ""))

    def add_prompt(self, role: str, prompt: str):
        """Add or update a prompt for a role.

        Args:
            role: The agent role
            prompt: The prompt text
        """
        self.prompts[role] = prompt

    def remove_prompt(self, role: str):
        """Remove a prompt for a role.

        Args:
            role: The agent role to remove
        """
        if role in self.prompts:
            del self.prompts[role]

    def list_roles(self) -> list[str]:
        """Get a list of all available roles.

        Returns:
            List[str]: List of role names
        """
        return list(self.prompts.keys())

    def has_role(self, role: str) -> bool:
        """Check if a role exists in the library.

        Args:
            role: The role to check

        Returns:
            bool: True if the role exists
        """
        return role in self.prompts

    def update_prompts(self, prompts: dict[str, str]):
        """Update multiple prompts at once.

        Args:
            prompts: Dictionary of role -> prompt mappings
        """
        self.prompts.update(prompts)

    def save_to_file(self, filepath: str):
        """Save all prompts to a YAML file.

        Args:
            filepath: Path to save the prompts
        """
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.prompts, f, default_flow_style=False, allow_unicode=True)

    def load_from_file(self, filepath: str):
        """Load prompts from a YAML file.

        Args:
            filepath: Path to load prompts from
        """
        with open(filepath, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                self.prompts.update(data)
