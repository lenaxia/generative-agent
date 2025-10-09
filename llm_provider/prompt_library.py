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
        """Load default prompts for common agent roles.

        Note: This method is deprecated. Prompts should now be defined in
        role YAML definitions and accessed via the RoleRegistry.
        """
        # Only keep a basic fallback prompt
        self.prompts.update(
            {
                "default": """You are a helpful AI assistant. Provide clear, accurate, and helpful responses to user queries.""",
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
