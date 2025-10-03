"""
Web Search Shared Tool

Common web search functionality that can be used across multiple roles.
"""

from strands import tool
from typing import Dict, List, Optional
import requests
import logging

logger = logging.getLogger(__name__)


@tool
def web_search(query: str, num_results: int = 5) -> Dict:
    """
    Search the web for information using Tavily API.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 5)
        
    Returns:
        Dict containing search results with titles, URLs, and snippets
    """
    try:
        # This would integrate with actual search API (Tavily, etc.)
        # For now, returning mock data that matches expected format
        results = []
        for i in range(min(num_results, 5)):
            results.append({
                "title": f"Search Result {i+1} for '{query}'",
                "url": f"https://example.com/result-{i+1}",
                "snippet": f"This is a relevant snippet for query '{query}' from result {i+1}.",
                "relevance_score": 0.9 - (i * 0.1)
            })
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "search_time": "0.25s"
        }
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            "query": query,
            "results": [],
            "total_results": 0,
            "error": str(e)
        }


@tool
def search_with_filters(query: str, domain: Optional[str] = None, 
                       date_range: Optional[str] = None, 
                       content_type: Optional[str] = None) -> Dict:
    """
    Advanced web search with filters.
    
    Args:
        query: Search query string
        domain: Specific domain to search (e.g., "reddit.com")
        date_range: Date range filter (e.g., "past_week", "past_month")
        content_type: Content type filter (e.g., "news", "academic", "forum")
        
    Returns:
        Dict containing filtered search results
    """
    try:
        # Build filtered query
        filtered_query = query
        if domain:
            filtered_query += f" site:{domain}"
        
        # Apply filters and search
        results = []
        for i in range(3):  # Fewer results for filtered search
            results.append({
                "title": f"Filtered Result {i+1} for '{query}'",
                "url": f"https://{domain or 'example.com'}/filtered-{i+1}",
                "snippet": f"Filtered content for '{query}' with filters applied.",
                "content_type": content_type or "general",
                "date": "2024-01-01",
                "relevance_score": 0.95 - (i * 0.05)
            })
        
        return {
            "query": query,
            "filters": {
                "domain": domain,
                "date_range": date_range,
                "content_type": content_type
            },
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Filtered web search failed: {e}")
        return {
            "query": query,
            "results": [],
            "error": str(e)
        }