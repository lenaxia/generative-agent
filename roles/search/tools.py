"""Search Domain Tools

Provides web and news search tools using Tavily Search API.
All tools are query-only (read search results).

Extracted from: roles/core_search.py
"""

import logging
import os
from typing import Any

import requests
from strands import tool

logger = logging.getLogger(__name__)


def create_search_tools(search_provider: Any) -> list:
    """Create search domain tools.

    Args:
        search_provider: Search provider instance (Tavily API)

    Returns:
        List of tool functions for search domain
    """
    tools = [
        web_search,
        search_news,
    ]

    logger.info(f"Created {len(tools)} search tools")
    return tools


# QUERY TOOLS (read-only)


@tool
def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web using Tavily Search API.

    Query tool - performs web search, no side effects.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Dict with success status, answer, and search results
    """
    logger.info(f"Performing web search for: {query}")

    try:
        # Get Tavily API key from environment
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": "Tavily API key not configured. Set TAVILY_API_KEY environment variable.",
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

        logger.info(f"Web search completed: {len(results)} results for '{query}'")
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
    """Search for recent news using Tavily Search API.

    Query tool - performs news search, no side effects.

    Args:
        query: News search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Dict with success status, answer, and news results
    """
    logger.info(f"Performing news search for: {query}")

    try:
        # Get Tavily API key from environment
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": "Tavily API key not configured. Set TAVILY_API_KEY environment variable.",
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
            "days": 7,  # Focus on recent news (last 7 days)
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
                    "score": result.get("score", 0),
                    "published_date": result.get("published_date"),
                }
            )

        logger.info(f"News search completed: {len(results)} results for '{query}'")
        return {
            "success": True,
            "query": query,
            "answer": search_data.get("answer", ""),
            "results": results,
            "total_results": len(results),
            "search_metadata": {
                "search_depth": "basic",
                "max_results": max_results,
                "days": 7,
            },
        }

    except Exception as e:
        logger.error(f"News search error for '{query}': {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
        }
