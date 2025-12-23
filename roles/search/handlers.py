"""Search Role Handlers - Event handlers and intent processors.

This module contains:
- SearchIntent and NewsSearchIntent definitions
- Event handlers (pure functions returning intents)
- Intent processors (async functions executing side effects)
- Helper functions for event parsing

Architecture: Intent-based event processing with LLM-safe patterns
Created: 2025-12-22 (Migrated from core_search.py)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)


# 1. ROLE-SPECIFIC INTENTS (owned by search role)
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


# 2. EVENT HANDLERS (pure functions returning intents)
def handle_search_request(event_data: Any, context) -> list[Intent]:
    """LLM-SAFE: Pure function for search request events.

    Args:
        event_data: Search request data (dict or list with query and max_results)
        context: LLM-safe event context

    Returns:
        List of intents: [SearchIntent, AuditIntent] or [NotificationIntent] on error
    """
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


# 3. HELPER FUNCTIONS (minimal, focused)
def _parse_search_event_data(event_data: Any) -> tuple[str, int]:
    """LLM-SAFE: Parse search event data with error handling.

    Args:
        event_data: Event data (dict or list)

    Returns:
        Tuple of (query, max_results)
    """
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
    """LLM-SAFE: Get safe channel from context.

    Args:
        context: Event context object

    Returns:
        Channel ID string or "console" as fallback
    """
    if hasattr(context, "get_safe_channel") and callable(context.get_safe_channel):
        try:
            return context.get_safe_channel()
        except TypeError:
            pass
    if hasattr(context, "channel_id") and context.channel_id:
        return context.channel_id
    else:
        return "console"


# 4. INTENT PROCESSORS (async functions executing side effects)
async def process_search_intent(intent: SearchIntent):
    """Process search intents - called by IntentProcessor.

    This is a placeholder that logs the intent. In a full implementation:
    - Would execute the search via Tavily API
    - Would format results for user consumption
    - Would handle API errors gracefully

    Note: The actual search happens via the web_search tool in tools.py,
    which is called by the UniversalAgent. This processor is for event-driven
    search requests that come through the MessageBus.

    Args:
        intent: SearchIntent with query and search parameters
    """
    logger.info(f"Processing search intent: {intent.query}")
    # Placeholder - actual search happens via tools in agent execution


async def process_news_search_intent(intent: NewsSearchIntent):
    """Process news search intents - called by IntentProcessor.

    This is a placeholder that logs the intent. In a full implementation:
    - Would execute news search via Tavily API
    - Would format news results with dates
    - Would handle API errors gracefully

    Note: The actual news search happens via the search_news tool in tools.py,
    which is called by the UniversalAgent. This processor is for event-driven
    news search requests that come through the MessageBus.

    Args:
        intent: NewsSearchIntent with query and search parameters
    """
    logger.info(f"Processing news search intent: {intent.query}")
    # Placeholder - actual news search happens via tools in agent execution


# 5. CONSTANTS AND CONFIGURATION
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
    """Normalize search action to standard form.

    Args:
        action: Action string from user request

    Returns:
        Normalized action name
    """
    return SEARCH_ACTIONS.get(action.lower(), "web_search")


# 6. ERROR HANDLING UTILITIES
def create_search_error_intent(error: Exception, context) -> list[Intent]:
    """Create error intents for search operations.

    Args:
        error: Exception that occurred
        context: Event context

    Returns:
        List of error intents: [NotificationIntent, AuditIntent]
    """
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
