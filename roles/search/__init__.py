"""Search Domain Package

Provides web search and news search capabilities using Tavily Search API.

This domain follows Phase 3 architecture:
- role.py: SearchRole class with configuration
- handlers.py: Event handlers and intent processors
- tools.py: Search tools (web_search, search_news)

Usage:
    from roles.search import SearchRole
    from roles.search.tools import create_search_tools
    from roles.search.handlers import SearchIntent, NewsSearchIntent
"""

from roles.search.handlers import NewsSearchIntent, SearchIntent
from roles.search.role import SearchRole
from roles.search.tools import create_search_tools

__all__ = [
    "SearchRole",
    "create_search_tools",
    "SearchIntent",
    "NewsSearchIntent",
]
