"""Search role - LLM-friendly single file implementation with Tavily Search API.

This role provides web search capabilities using the Tavily Search API,
following the single-file role architecture pattern.

Key architectural principles:
- Single event loop compliance
- Intent-based processing (pure functions returning intents)
- LLM-safe patterns (predictable, simple, self-contained)
- Tavily Search API integration for web search

Architecture: Single Event Loop + Intent-Based + Tavily API
Created: 2025-10-18
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
from strands import tool

from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA
ROLE_CONFIG = {
    "name": "search",
    "version": "1.0.0",
    "description": "Web search using Tavily Search API with LLM-safe architecture",
    "llm_type": "DEFAULT",
    "fast_reply": True,
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
    "tools": {
        "automatic": True,  # Include search tools
        "shared": [],  # No shared tools needed
        "include_builtin": False,  # No built-in tools
        "fast_reply": {
            "enabled": True,  # Enable tools in fast-reply mode
        },
    },
    "prompts": {
        "system": """You are a web search specialist providing current information from the internet.

Available search tools:
- web_search(query, max_results): Search the web for information on any topic
- search_news(query, max_results): Search for recent news articles on a topic

When users request information:
1. Determine the best search query from their request
2. Use the appropriate search tool
3. Provide a comprehensive summary of the findings
4. Include relevant sources and links when available

Always use the search tools to get current information. Provide clear, informative responses based on the search results."""
    },
}


# 2. ROLE-SPECIFIC INTENTS (owned by search role)
@dataclass
class SearchIntent(Intent):
    """Search-specific intent: Execute web search with Tavily API."""

    query: str
    max_results: int = 5
    search_depth: str = "basic"
    user_id: str | None = None
    channel_id: str | None = None

    def validate(self) -> bool:
        """Validate search intent parameters."""
        return (
            bool(self.query and len(self.query.strip()) > 0)
            and isinstance(self.max_results, int)
            and self.max_results > 0
            and self.search_depth in ["basic", "advanced"]
        )


@dataclass
class NewsSearchIntent(Intent):
    """News search intent: Search for recent news articles."""

    query: str
    max_results: int = 5
    user_id: str | None = None
    channel_id: str | None = None

    def validate(self) -> bool:
        """Validate news search intent parameters."""
        return (
            bool(self.query and len(self.query.strip()) > 0)
            and isinstance(self.max_results, int)
            and self.max_results > 0
        )


# 3. EVENT HANDLERS (pure functions returning intents)
def handle_search_request(event_data: Any, context) -> list[Intent]:
    """LLM-SAFE: Pure function for search request events."""
    try:
        # Parse event data
        query, max_results = _parse_search_event_data(event_data)

        # Check if parsing failed
        if query == "parse_error":
            logger.error(f"Search handler error: Invalid event data: {event_data}")
            return [
                NotificationIntent(
                    message="Search processing error: Invalid event data",
                    channel=_get_safe_channel(context),
                    priority="high",
                    notification_type="error",
                )
            ]

        # Create intents for successful parsing
        return [
            SearchIntent(
                query=query,
                max_results=max_results,
                search_depth="basic",
                user_id=getattr(context, "user_id", None),
                channel_id=getattr(context, "channel_id", None),
            ),
            AuditIntent(
                action="search_request",
                details={
                    "query": query,
                    "max_results": max_results,
                    "processed_at": time.time(),
                },
                user_id=getattr(context, "user_id", None),
            ),
        ]

    except Exception as e:
        logger.error(f"Search handler error: {e}")
        return [
            NotificationIntent(
                message=f"Search processing error: {e}",
                channel=_get_safe_channel(context),
                priority="high",
                notification_type="error",
            )
        ]


# 4. TOOLS (declarative, LLM-friendly, intent-based)
@tool
def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """LLM-SAFE: Search the web using Tavily Search API."""
    try:
        logger.info(f"Performing web search for: {query}")

        # Get Tavily API key from environment
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": "Tavily API key not configured",
                "query": query,
            }

        # Tavily Search API request
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_images": False,
            "include_raw_content": False,
            "max_results": max_results,
        }

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        search_data = response.json()

        # Format results
        results = []
        for result in search_data.get("results", []):
            results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0),
                }
            )

        return {
            "success": True,
            "query": query,
            "answer": search_data.get("answer", ""),
            "results": results,
            "total_results": len(results),
            "search_metadata": {
                "search_depth": "basic",
                "max_results": max_results,
            },
        }

    except Exception as e:
        logger.error(f"Web search error for '{query}': {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
        }


@tool
def search_news(query: str, max_results: int = 5) -> dict[str, Any]:
    """LLM-SAFE: Search for recent news using Tavily Search API."""
    try:
        logger.info(f"Performing news search for: {query}")

        # Get Tavily API key from environment
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": "Tavily API key not configured",
                "query": query,
            }

        # Tavily Search API request with news focus
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": f"{query} news recent",
            "search_depth": "basic",
            "include_answer": True,
            "include_images": False,
            "include_raw_content": False,
            "max_results": max_results,
            "days": 7,  # Focus on recent news
        }

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        search_data = response.json()

        # Format news results
        results = []
        for result in search_data.get("results", []):
            results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "published_date": result.get("published_date", ""),
                    "score": result.get("score", 0),
                }
            )

        return {
            "success": True,
            "query": query,
            "answer": search_data.get("answer", ""),
            "results": results,
            "total_results": len(results),
            "search_metadata": {
                "search_type": "news",
                "search_depth": "basic",
                "max_results": max_results,
                "days_back": 7,
            },
        }

    except Exception as e:
        logger.error(f"News search error for '{query}': {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
        }


# 5. HELPER FUNCTIONS (minimal, focused)
def _parse_search_event_data(event_data: Any) -> tuple[str, int]:
    """LLM-SAFE: Parse search event data with error handling."""
    try:
        if isinstance(event_data, dict):
            return (
                event_data.get("query", "unknown"),
                event_data.get("max_results", 5),
            )
        elif isinstance(event_data, list) and len(event_data) >= 1:
            return str(event_data[0]), int(event_data[1]) if len(event_data) > 1 else 5
        else:
            return "unknown", 5
    except Exception as e:
        return "parse_error", 5


def _get_safe_channel(context) -> str:
    """LLM-SAFE: Get safe channel from context."""
    if hasattr(context, "get_safe_channel") and callable(context.get_safe_channel):
        try:
            return context.get_safe_channel()
        except TypeError:
            pass
    if hasattr(context, "channel_id") and context.channel_id:
        return context.channel_id
    else:
        return "console"


# 6. INTENT HANDLER REGISTRATION
async def process_search_intent(intent: SearchIntent):
    """Process search intents - handles actual Tavily API calls."""
    logger.info(f"Processing search intent: {intent.query}")

    # In full implementation, this would:
    # - Execute the search via Tavily API
    # - Format results for user consumption
    # - Handle API errors gracefully
    # For now, just log the intent processing


async def process_news_search_intent(intent: NewsSearchIntent):
    """Process news search intents - handles actual news search."""
    logger.info(f"Processing news search intent: {intent.query}")

    # In full implementation, this would:
    # - Execute news search via Tavily API
    # - Format news results with dates
    # - Handle API errors gracefully
    # For now, just log the intent processing


# 7. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "SEARCH_REQUEST": handle_search_request,
        },
        "tools": [web_search, search_news],
        "intents": {
            SearchIntent: process_search_intent,
            NewsSearchIntent: process_news_search_intent,
        },
    }


# 8. CONSTANTS AND CONFIGURATION
TAVILY_API_URL = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 5
MAX_SEARCH_RESULTS = 20
SEARCH_TIMEOUT = 10  # seconds

# Search action mappings for LLM understanding
SEARCH_ACTIONS = {
    "search": "web_search",
    "find": "web_search",
    "lookup": "web_search",
    "research": "web_search",
    "news": "search_news",
    "recent": "search_news",
    "latest": "search_news",
}


def normalize_search_action(action: str) -> str:
    """Normalize search action to standard form."""
    return SEARCH_ACTIONS.get(action.lower(), "web_search")


# 9. ERROR HANDLING UTILITIES
def create_search_error_intent(error: Exception, context) -> list[Intent]:
    """Create error intents for search operations."""
    return [
        NotificationIntent(
            message=f"Search error: {error}",
            channel=_get_safe_channel(context),
            user_id=getattr(context, "user_id", None),
            priority="high",
            notification_type="error",
        ),
        AuditIntent(
            action="search_error",
            details={"error": str(error), "context": str(context)},
            user_id=getattr(context, "user_id", None),
            severity="error",
        ),
    ]
