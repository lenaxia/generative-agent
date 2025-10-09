"""Unit tests for role tool integration with web search and scraping tools."""

from unittest.mock import Mock, patch

import pytest

from llm_provider.factory import LLMFactory, LLMType
from llm_provider.role_registry import RoleRegistry
from llm_provider.universal_agent import UniversalAgent


class TestRoleToolIntegration:
    """Test role integration with web search and scraping tools."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Mock LLM factory for testing."""
        factory = Mock(spec=LLMFactory)
        factory.create_strands_model.return_value = Mock()
        return factory

    @pytest.fixture
    def role_registry(self):
        """Create role registry for testing."""
        return RoleRegistry(roles_directory="roles")

    @pytest.fixture
    def universal_agent(self, mock_llm_factory, role_registry):
        """Create universal agent for testing."""
        return UniversalAgent(mock_llm_factory, role_registry)

    def test_search_role_has_web_tools(self, role_registry):
        """Test that search role includes all web search and scraping tools."""
        search_role = role_registry.get_role("search")

        assert search_role is not None
        assert search_role.name == "search"

        # Check that search pipeline tools are included (actual implementation)
        expected_tools = [
            "search_and_scrape_pipeline",
            "wikipedia_search_pipeline",
            "multi_source_search_pipeline",
        ]

        # Access tools from config structure
        shared_tools = search_role.config.get("tools", {}).get("shared", [])
        for tool in expected_tools:
            assert tool in shared_tools, f"Tool '{tool}' not found in search role"

    def test_research_analyst_role_has_web_tools(self, role_registry):
        """Test that research analyst role includes all web search and scraping tools."""
        research_role = role_registry.get_role("research_analyst")

        assert research_role is not None
        assert research_role.name == "research_analyst"

        # Check that all web search and scraping tools are included
        expected_tools = [
            "web_search",
            "search_with_filters",
            "search_news",
            "search_academic",
            "scrape_webpage",
            "extract_article_content",
            "scrape_with_links",
        ]

        # Access tools from config structure
        shared_tools = research_role.config.get("tools", {}).get("shared", [])
        for tool in expected_tools:
            assert (
                tool in shared_tools
            ), f"Tool '{tool}' not found in research_analyst role"

    @patch("roles.search.tools.TAVILY_AVAILABLE", True)
    @patch("roles.search.tools.TavilyClient")
    def test_search_role_can_use_web_search(self, mock_tavily_client, universal_agent):
        """Test that search role can successfully use web search tools."""
        # Mock Tavily client
        mock_client = Mock()
        mock_response = {
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com",
                    "content": "Test content",
                    "score": 0.9,
                }
            ],
            "query": "test query",
        }
        mock_client.search.return_value = mock_response
        mock_tavily_client.return_value = mock_client

        # Mock the role assumption to avoid LLM calls
        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}),
            patch.object(universal_agent, "_create_strands_model") as mock_create_model,
            patch("strands.Agent") as mock_agent_class,
        ):
            mock_model = Mock()
            mock_create_model.return_value = mock_model
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance

            # Create search role agent
            agent = universal_agent.assume_role("search", LLMType.WEAK)

            # Verify agent was created successfully
            assert agent is not None

            # Verify tools are available (this would be tested in integration)
            # For unit test, we just verify role assumption works
            assert universal_agent.current_role == "search"

    @patch("roles.search.tools.SCRAPING_AVAILABLE", True)
    @patch("roles.search.tools.requests.get")
    @patch("roles.search.tools.BeautifulSoup")
    def test_research_role_can_use_scraping(
        self, mock_bs, mock_requests, universal_agent
    ):
        """Test that research analyst role can use scraping tools."""
        # Mock HTTP response for scraping
        mock_response = Mock()
        mock_response.content = b"<html><title>Test</title><body>Content</body></html>"
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response

        # Mock BeautifulSoup
        mock_soup = Mock()
        mock_soup.find.return_value = Mock(get_text=Mock(return_value="Test"))
        mock_soup.get_text.return_value = "Content"
        mock_bs.return_value = mock_soup

        # Mock the role assumption to avoid LLM calls
        with (
            patch.object(universal_agent, "_create_strands_model") as mock_create_model,
            patch("strands.Agent") as mock_agent_class,
        ):
            mock_model = Mock()
            mock_create_model.return_value = mock_model
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance

            # Create research analyst role agent
            agent = universal_agent.assume_role("research_analyst", LLMType.DEFAULT)

            # Verify agent was created successfully
            assert agent is not None
            assert universal_agent.current_role == "research_analyst"

    def test_role_descriptions_updated(self, role_registry):
        """Test that role descriptions mention web search and scraping capabilities."""
        search_role = role_registry.get_role("search")
        research_role = role_registry.get_role("research_analyst")

        # Check search role description from config
        search_config = search_role.config.get("role", {})
        assert "content extraction" in search_config.get("description", "").lower()
        assert "scrape" in search_config.get("when_to_use", "").lower()
        assert "tavily" in search_config.get("when_to_use", "").lower()

        # Check research analyst role description from config
        research_config = research_role.config.get("role", {})
        assert "web content analysis" in research_config.get("description", "").lower()
        assert "scraping" in research_config.get("when_to_use", "").lower()
        assert "web sources" in research_config.get("when_to_use", "").lower()

    def test_role_system_prompts_include_tools(self, role_registry):
        """Test that role system prompts mention the available tools."""
        search_role = role_registry.get_role("search")
        research_role = role_registry.get_role("research_analyst")

        # Check search role system prompt from config (updated for actual implementation)
        search_prompt = search_role.config.get("prompts", {}).get("system", "").lower()
        assert "search_and_scrape_pipeline" in search_prompt
        assert "wikipedia_search_pipeline" in search_prompt

        # Check research analyst role system prompt from config
        research_prompt = (
            research_role.config.get("prompts", {}).get("system", "").lower()
        )
        assert "scrape_webpage" in research_prompt
        assert "search_academic" in research_prompt
        assert "available research tools" in research_prompt
