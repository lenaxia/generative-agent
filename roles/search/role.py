"""Search Role - Domain-based implementation with Tavily Search API.

This role provides web search capabilities using the Tavily Search API,
following the Phase 3 domain-based architecture pattern.

Key architectural principles:
- Domain-based organization (role.py, handlers.py, tools.py)
- Dependency injection (ToolRegistry, LLMFactory)
- Intent-based processing (pure functions returning intents)
- LLM-safe patterns (predictable, simple, self-contained)

Architecture: Phase 3 Domain Role + Intent-Based + Tavily API
Created: 2025-12-22 (Migrated from core_search.py)
"""

import logging
from typing import Any

from llm_provider.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class SearchRole:
    """Search domain role for web and news search using Tavily API.

    This role provides fast-reply web search capabilities with:
    - Tavily Search API integration
    - Web search and news search functionality
    - Fast response times for information retrieval

    Fast-reply role: Single-purpose information retrieval without multi-step workflows.
    """

    # Required tools for this role (loaded from ToolRegistry)
    REQUIRED_TOOLS = [
        "search.web_search",
        "search.search_news",
    ]

    def __init__(self, tool_registry: ToolRegistry, llm_factory: Any):
        """Initialize the search role.

        Args:
            tool_registry: Central tool registry for loading domain tools
            llm_factory: Factory for creating LLM instances
        """
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.tools = []

        logger.info("SearchRole initialized")

    async def initialize(self):
        """Initialize the search role and load tools from registry."""
        # Load search tools from tool registry using REQUIRED_TOOLS
        self.tools = self.tool_registry.get_tools(self.REQUIRED_TOOLS)

        # Count how many tools were successfully loaded
        loaded_count = len(self.tools)
        required_count = len(self.REQUIRED_TOOLS)

        if loaded_count < required_count:
            logger.warning(
                f"SearchRole: {required_count - loaded_count} tools could not be loaded"
            )

        logger.info(
            f"SearchRole loaded {loaded_count} tools: "
            f"{[self.tool_registry._extract_tool_name(t) for t in self.tools]}"
        )

    def get_system_prompt(self) -> str:
        """Get the system prompt for the search role.

        Returns:
            System prompt string defining search role behavior
        """
        return """You are a web search specialist providing current information from the internet.

Available search tools:
- web_search(query, max_results): Search the web for information on any topic
- search_news(query, max_results): Search for recent news articles on a topic

When users request information:
1. Determine the best search query from their request
2. Use the appropriate search tool
3. Provide a comprehensive summary of the findings
4. Include relevant sources and links when available

Always use the search tools to get current information. Provide clear, informative responses based on the search results."""

    def get_llm_type(self) -> str:
        """Get the LLM type for this role.

        Returns:
            LLM type identifier
        """
        return "DEFAULT"

    def get_tools(self) -> list[Any]:
        """Get tools for this role.

        Returns:
            List of tool objects
        """
        return self.tools

    def get_role_config(self) -> dict:
        """Get role configuration for registry.

        Returns:
            Dictionary with role configuration including fast_reply flag
        """
        return {
            "name": "search",
            "version": "1.0.0",
            "description": "Web search using Tavily Search API with LLM-safe architecture",
            "llm_type": "DEFAULT",
            "fast_reply": True,  # Fast-reply role for quick information retrieval
            "when_to_use": "Search the web for current information, research topics, find recent news or data",
            "parameters": {
                "query": {
                    "type": "string",
                    "required": True,
                    "description": "Search query to execute",
                    "examples": [
                        "latest AI developments",
                        "weather in Seattle",
                        "Python tutorials",
                    ],
                },
                "max_results": {
                    "type": "integer",
                    "required": False,
                    "description": "Maximum number of search results to return",
                    "examples": [5, 10, 20],
                    "default": 5,
                },
                "search_depth": {
                    "type": "string",
                    "required": False,
                    "description": "Depth of search results",
                    "examples": ["basic", "advanced"],
                    "enum": ["basic", "advanced"],
                    "default": "basic",
                },
            },
        }

    def get_event_handlers(self):
        """Get event handlers for this role.

        Returns:
            Dictionary mapping event types to handler functions
        """
        from roles.search.handlers import handle_search_request

        return {
            "SEARCH_REQUEST": handle_search_request,
        }

    def get_intent_handlers(self):
        """Get intent handlers for this role.

        Returns:
            Dictionary mapping Intent classes to processor functions
        """
        from roles.search.handlers import (
            NewsSearchIntent,
            SearchIntent,
            process_news_search_intent,
            process_search_intent,
        )

        return {
            SearchIntent: process_search_intent,
            NewsSearchIntent: process_news_search_intent,
        }
