"""
Search tools for StrandsAgent - converted from SearchAgent.

These tools replace the LangChain-based SearchAgent with @tool decorated functions
that can be used by the Universal Agent for web search functionality.
"""

import requests
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def web_search(query: str, num_results: int = 5, search_depth: str = "basic") -> Dict[str, Any]:
    """
    Search the web for information - converted from SearchAgent.
    
    This tool performs web search using available search APIs and returns
    structured results with titles, URLs, and content snippets.
    
    Args:
        query: The search query string
        num_results: Maximum number of results to return (default: 5)
        search_depth: Search depth - "basic" or "advanced" (default: "basic")
        
    Returns:
        Dict containing search results with titles, URLs, and snippets
    """
    logger.info(f"Performing web search for query: {query}")
    
    # For now, return mock search results since we don't have Tavily API key
    # In a real implementation, this would use a search API like Tavily, Bing, or Google
    mock_results = [
        {
            "title": f"Search Result 1 for '{query}'",
            "url": "https://example.com/result1",
            "content": f"This is a mock search result for the query '{query}'. In a real implementation, this would contain actual web search results.",
            "published_date": datetime.now().isoformat(),
            "score": 0.95
        },
        {
            "title": f"Search Result 2 for '{query}'",
            "url": "https://example.com/result2", 
            "content": f"Another mock search result providing information about '{query}'. This would be replaced with real search API results.",
            "published_date": datetime.now().isoformat(),
            "score": 0.87
        }
    ]
    
    # Limit results to requested number
    results = mock_results[:num_results]
    
    search_result = {
        "query": query,
        "num_results": len(results),
        "search_depth": search_depth,
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }
    
    logger.info(f"Web search completed: {len(results)} results for '{query}'")
    return search_result


def search_with_filters(query: str, filters: Dict[str, Any] = None, num_results: int = 5) -> Dict[str, Any]:
    """
    Search the web with additional filters.
    
    Args:
        query: The search query string
        filters: Dictionary of search filters (domain, date_range, content_type, etc.)
        num_results: Maximum number of results to return
        
    Returns:
        Dict containing filtered search results
    """
    logger.info(f"Performing filtered web search for query: {query} with filters: {filters}")
    
    filters = filters or {}
    
    # Apply filters to search (mock implementation)
    base_result = web_search(query, num_results)
    
    # Add filter information to result
    base_result["filters_applied"] = filters
    base_result["filtered"] = True
    
    logger.info(f"Filtered search completed: {len(base_result['results'])} results")
    return base_result


def search_news(query: str, num_results: int = 3, days_back: int = 7) -> Dict[str, Any]:
    """
    Search for recent news articles.
    
    Args:
        query: The search query string
        num_results: Maximum number of news results to return
        days_back: How many days back to search for news
        
    Returns:
        Dict containing news search results
    """
    logger.info(f"Performing news search for query: {query} (last {days_back} days)")
    
    # Mock news results
    news_results = [
        {
            "title": f"Breaking News: {query}",
            "url": "https://news.example.com/breaking",
            "content": f"Recent news about {query}. This would contain actual news content from news APIs.",
            "published_date": datetime.now().isoformat(),
            "source": "Example News",
            "category": "general"
        }
    ]
    
    results = news_results[:num_results]
    
    news_result = {
        "query": query,
        "num_results": len(results),
        "days_back": days_back,
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "search_type": "news",
        "status": "success"
    }
    
    logger.info(f"News search completed: {len(results)} results")
    return news_result


def search_academic(query: str, num_results: int = 3) -> Dict[str, Any]:
    """
    Search for academic papers and scholarly content.
    
    Args:
        query: The search query string
        num_results: Maximum number of academic results to return
        
    Returns:
        Dict containing academic search results
    """
    logger.info(f"Performing academic search for query: {query}")
    
    # Mock academic results
    academic_results = [
        {
            "title": f"Academic Paper on {query}",
            "url": "https://scholar.example.com/paper1",
            "content": f"Abstract: This paper discusses {query} and its implications...",
            "authors": ["Dr. Smith", "Dr. Johnson"],
            "published_date": "2023-01-15",
            "journal": "Journal of Example Studies",
            "citations": 42,
            "doi": "10.1000/example.doi"
        }
    ]
    
    results = academic_results[:num_results]
    
    academic_result = {
        "query": query,
        "num_results": len(results),
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "search_type": "academic",
        "status": "success"
    }
    
    logger.info(f"Academic search completed: {len(results)} results")
    return academic_result


def summarize_search_results(search_results: Dict[str, Any], max_length: int = 200) -> Dict[str, Any]:
    """
    Summarize search results into a concise summary.
    
    Args:
        search_results: Results from web_search or other search functions
        max_length: Maximum length of summary in words
        
    Returns:
        Dict containing the summary and metadata
    """
    logger.info(f"Summarizing search results for query: {search_results.get('query', 'unknown')}")
    
    if not search_results.get("results"):
        return {
            "summary": "No search results to summarize.",
            "query": search_results.get("query", ""),
            "num_results_summarized": 0,
            "status": "no_results"
        }
    
    # Create a simple summary from the search results
    query = search_results.get("query", "")
    results = search_results.get("results", [])
    
    summary_parts = [f"Search results for '{query}':"]
    
    for i, result in enumerate(results[:3], 1):  # Summarize top 3 results
        title = result.get("title", "")
        content = result.get("content", "")[:100]  # First 100 chars
        summary_parts.append(f"{i}. {title}: {content}...")
    
    summary = " ".join(summary_parts)
    
    # Truncate to max_length words
    words = summary.split()
    if len(words) > max_length:
        summary = " ".join(words[:max_length]) + "..."
    
    result = {
        "summary": summary,
        "query": query,
        "num_results_summarized": len(results),
        "max_length": max_length,
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }
    
    logger.info(f"Search results summarized: {len(words)} words")
    return result


def validate_search_query(query: str) -> Dict[str, Any]:
    """
    Validate and suggest improvements for search queries.
    
    Args:
        query: The search query to validate
        
    Returns:
        Dict containing validation results and suggestions
    """
    logger.info(f"Validating search query: {query}")
    
    issues = []
    suggestions = []
    
    # Basic validation
    if not query or not query.strip():
        issues.append("Query is empty")
        suggestions.append("Provide a non-empty search query")
    elif len(query.strip()) < 3:
        issues.append("Query is too short")
        suggestions.append("Use at least 3 characters for better search results")
    elif len(query) > 200:
        issues.append("Query is very long")
        suggestions.append("Consider shortening the query for better results")
    
    # Check for common issues
    if query.count('"') % 2 != 0:
        issues.append("Unmatched quotes in query")
        suggestions.append("Ensure all quotes are properly paired")
    
    is_valid = len(issues) == 0
    
    result = {
        "query": query,
        "valid": is_valid,
        "issues": issues,
        "suggestions": suggestions,
        "estimated_quality": "high" if is_valid and len(query.split()) > 2 else "medium" if is_valid else "low"
    }
    
    logger.info(f"Query validation: {'PASSED' if is_valid else 'FAILED'} with {len(issues)} issues")
    return result


# Tool registry for search tools
SEARCH_TOOLS = {
    "web_search": web_search,
    "search_with_filters": search_with_filters,
    "search_news": search_news,
    "search_academic": search_academic,
    "summarize_search_results": summarize_search_results,
    "validate_search_query": validate_search_query
}


def get_search_tools() -> Dict[str, Any]:
    """Get all available search tools."""
    return SEARCH_TOOLS


def get_search_tool_descriptions() -> Dict[str, str]:
    """Get descriptions of all search tools."""
    return {
        "web_search": "Search the web for general information",
        "search_with_filters": "Search the web with additional filters",
        "search_news": "Search for recent news articles",
        "search_academic": "Search for academic papers and scholarly content",
        "summarize_search_results": "Summarize search results into a concise summary",
        "validate_search_query": "Validate and suggest improvements for search queries"
    }