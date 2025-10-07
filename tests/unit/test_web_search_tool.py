"""
Unit tests for web search tool with Tavily API integration.
"""

import os
from unittest.mock import Mock, patch

from roles.shared_tools.web_search import search_with_filters, web_search


class TestWebSearchTool:
    """Test cases for web search functionality."""

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test_key"})
    @patch("roles.shared_tools.web_search.TAVILY_AVAILABLE", True)
    @patch("roles.shared_tools.web_search.TavilyClient")
    def test_web_search_success(self, mock_tavily_client):
        """Test successful web search with Tavily API."""
        # Mock Tavily client and response
        mock_client = Mock()
        mock_tavily_client.return_value = mock_client

        mock_response = {
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "content": "This is test content for result 1",
                    "score": 0.95,
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "content": "This is test content for result 2",
                    "score": 0.87,
                },
            ],
            "query": "test query",
        }
        mock_client.search.return_value = mock_response

        # Test the function
        result = web_search("test query", num_results=2)

        # Verify API call
        mock_client.search.assert_called_once_with(
            query="test query",
            max_results=2,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
        )

        # Verify result structure
        assert result["query"] == "test query"
        assert result["total_results"] == 2
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Test Result 1"
        assert result["results"][0]["url"] == "https://example.com/1"
        assert result["results"][0]["snippet"] == "This is test content for result 1"
        assert result["results"][0]["relevance_score"] == 0.95

    @patch.dict(os.environ, {}, clear=True)
    def test_web_search_no_api_key(self):
        """Test web search when TAVILY_API_KEY is not set."""
        result = web_search("test query")

        assert result["query"] == "test query"
        assert result["total_results"] == 0
        assert "error" in result
        assert "TAVILY_API_KEY environment variable not set" in result["error"]

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test_key"})
    @patch("roles.shared_tools.web_search.TAVILY_AVAILABLE", True)
    @patch("roles.shared_tools.web_search.TavilyClient")
    def test_web_search_api_error(self, mock_tavily_client):
        """Test web search when Tavily API returns an error."""
        mock_client = Mock()
        mock_tavily_client.return_value = mock_client
        mock_client.search.side_effect = Exception("API Error")

        result = web_search("test query")

        assert result["query"] == "test query"
        assert result["total_results"] == 0
        assert "error" in result
        assert "API Error" in result["error"]

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test_key"})
    @patch("roles.shared_tools.web_search.TAVILY_AVAILABLE", True)
    @patch("roles.shared_tools.web_search.TavilyClient")
    def test_web_search_empty_results(self, mock_tavily_client):
        """Test web search when API returns empty results."""
        mock_client = Mock()
        mock_tavily_client.return_value = mock_client
        mock_client.search.return_value = {"results": [], "query": "test query"}

        result = web_search("test query")

        assert result["query"] == "test query"
        assert result["total_results"] == 0
        assert result["results"] == []

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test_key"})
    @patch("roles.shared_tools.web_search.TAVILY_AVAILABLE", True)
    @patch("roles.shared_tools.web_search.TavilyClient")
    def test_search_with_filters_success(self, mock_tavily_client):
        """Test filtered search with domain and content type."""
        mock_client = Mock()
        mock_tavily_client.return_value = mock_client

        mock_response = {
            "results": [
                {
                    "title": "Reddit Discussion",
                    "url": "https://reddit.com/r/test/post1",
                    "content": "Reddit discussion content",
                    "score": 0.92,
                }
            ],
            "query": "test query site:reddit.com",
        }
        mock_client.search.return_value = mock_response

        result = search_with_filters(
            query="test query", domain="reddit.com", content_type="forum"
        )

        # Verify the filtered query was constructed
        mock_client.search.assert_called_once_with(
            query="test query site:reddit.com",
            max_results=5,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
        )

        # Verify result structure
        assert result["query"] == "test query"
        assert result["filters"]["domain"] == "reddit.com"
        assert result["filters"]["content_type"] == "forum"
        assert result["total_results"] == 1
        assert result["results"][0]["title"] == "Reddit Discussion"

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test_key"})
    @patch("roles.shared_tools.web_search.TAVILY_AVAILABLE", True)
    @patch("roles.shared_tools.web_search.TavilyClient")
    def test_search_with_filters_date_range(self, mock_tavily_client):
        """Test filtered search with date range."""
        mock_client = Mock()
        mock_tavily_client.return_value = mock_client

        mock_response = {
            "results": [
                {
                    "title": "Recent News",
                    "url": "https://news.com/recent",
                    "content": "Recent news content",
                    "score": 0.88,
                }
            ],
            "query": "test query",
        }
        mock_client.search.return_value = mock_response

        result = search_with_filters(
            query="test query", date_range="past_week", content_type="news"
        )

        # Verify API call (date range handled by Tavily internally)
        mock_client.search.assert_called_once()

        # Verify result structure includes filters
        assert result["filters"]["date_range"] == "past_week"
        assert result["filters"]["content_type"] == "news"

    def test_web_search_parameter_validation(self):
        """Test parameter validation for web search."""
        # Test with invalid num_results
        result = web_search("test", num_results=0)
        assert result["total_results"] == 0

        # Test with empty query
        result = web_search("")
        assert result["query"] == ""
        assert result["total_results"] == 0

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test_key"})
    @patch("roles.shared_tools.web_search.TAVILY_AVAILABLE", True)
    @patch("roles.shared_tools.web_search.TavilyClient")
    def test_web_search_large_num_results(self, mock_tavily_client):
        """Test web search with large number of results."""
        mock_client = Mock()
        mock_tavily_client.return_value = mock_client

        # Create mock response with many results
        mock_results = []
        for i in range(20):
            mock_results.append(
                {
                    "title": f"Result {i+1}",
                    "url": f"https://example.com/{i+1}",
                    "content": f"Content for result {i+1}",
                    "score": 0.9 - (i * 0.01),
                }
            )

        mock_response = {"results": mock_results, "query": "test query"}
        mock_client.search.return_value = mock_response

        result = web_search("test query", num_results=20)

        # Verify API call with correct max_results
        mock_client.search.assert_called_once_with(
            query="test query",
            max_results=20,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
        )

        assert result["total_results"] == 20
        assert len(result["results"]) == 20
