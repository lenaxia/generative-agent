"""
Unit tests for SearchDataCollectorRole programmatic role.

Tests the SearchDataCollectorRole that performs pure data collection without analysis,
designed to eliminate redundant LLM analysis calls in search → analysis workflows.
"""

import json
from unittest.mock import Mock, call

import pytest

from common.task_context import TaskContext


class TestSearchDataCollectorRole:
    """Test suite for SearchDataCollectorRole programmatic role."""

    @pytest.fixture
    def mock_search_tools(self):
        """Create mock search tools."""
        tools = Mock()
        tools.search_source.return_value = [
            {
                "title": "Test Result 1",
                "url": "http://example1.com",
                "snippet": "Test content 1",
            },
            {
                "title": "Test Result 2",
                "url": "http://example2.com",
                "snippet": "Test content 2",
            },
        ]
        return tools

    @pytest.fixture
    def mock_llm_factory(self):
        """Create mock LLM factory for instruction parsing."""
        factory = Mock()
        mock_agent = Mock()
        mock_agent.return_value = json.dumps(
            {
                "query": "USS Monitor",
                "sources": ["web", "wikipedia"],
                "num_results": 5,
                "focus": "historical information",
            }
        )
        factory.create_agent.return_value = mock_agent
        return factory

    @pytest.fixture
    def search_data_collector_role(self, mock_search_tools, mock_llm_factory):
        """Create SearchDataCollectorRole instance with mocked dependencies."""
        # Import after implementation
        from roles.programmatic.search_data_collector_role import (
            SearchDataCollectorRole,
        )

        role = SearchDataCollectorRole()
        role.search_tools = mock_search_tools
        role.llm_factory = mock_llm_factory
        return role

    def test_search_data_collector_initialization(self):
        """Test SearchDataCollectorRole initialization."""
        from roles.programmatic.search_data_collector_role import (
            SearchDataCollectorRole,
        )

        role = SearchDataCollectorRole()

        assert role.name == "search_data_collector"
        assert (
            "collects" in role.description.lower()
            and "data" in role.description.lower()
        )
        assert role.execution_count == 0
        assert role.total_execution_time == 0.0

    def test_parse_instruction_with_llm(self, search_data_collector_role):
        """Test instruction parsing using LLM."""
        instruction = "Search for information about USS Monitor ironclad ship"

        params = search_data_collector_role._llm_parse_search_instruction(instruction)

        # Should return parsed parameters
        assert params["query"] == "USS Monitor"
        assert "web" in params["sources"]
        assert "wikipedia" in params["sources"]
        assert params["num_results"] == 5
        assert params["focus"] == "historical information"

        # Should have used WEAK model for cost optimization
        search_data_collector_role.llm_factory.create_agent.assert_called_once()

    def test_execute_search_pipeline_pure_data_collection(
        self, search_data_collector_role
    ):
        """Test pure data collection without analysis."""
        params = {
            "query": "USS Monitor",
            "sources": ["web", "wikipedia"],
            "num_results": 5,
        }

        results = search_data_collector_role._execute_search_pipeline(params)

        # Should return raw structured data without analysis
        assert len(results) == 4  # 2 results per source
        assert all("title" in result for result in results)
        assert all("url" in result for result in results)
        assert all("snippet" in result for result in results)

        # Should have called search for each source
        expected_calls = [
            call(query="USS Monitor", source="web", num_results=5),
            call(query="USS Monitor", source="wikipedia", num_results=5),
        ]
        search_data_collector_role.search_tools.search_source.assert_has_calls(
            expected_calls
        )

    def test_execute_full_workflow_without_analysis(self, search_data_collector_role):
        """Test complete execution workflow without LLM analysis."""
        instruction = "Search for USS Monitor information"
        mock_context = Mock(spec=TaskContext)

        result = search_data_collector_role.execute(instruction, mock_context)

        # Should return structured data without analysis
        assert "search_results" in result
        assert "metadata" in result
        assert "execution_type" in result
        assert result["execution_type"] == "programmatic_data_collection"

        # Should have search results
        assert len(result["search_results"]) > 0

        # Should have metadata
        metadata = result["metadata"]
        assert "query" in metadata
        assert "sources_searched" in metadata
        assert "total_results" in metadata
        assert "search_timestamp" in metadata
        assert "llm_calls" in metadata

        # Should have used minimal LLM calls (only for parsing)
        assert metadata["llm_calls"] <= 2  # 1 for parsing, maybe 1 for follow-up

    def test_needs_followup_logic(self, search_data_collector_role):
        """Test follow-up search decision logic."""
        # Test case where follow-up is needed (insufficient results)
        insufficient_results = [{"title": "Only one result"}]
        params = {"min_results": 3}

        assert (
            search_data_collector_role._needs_followup(insufficient_results, params)
            is True
        )

        # Test case where follow-up is not needed (sufficient results)
        sufficient_results = [
            {"title": "Result 1"},
            {"title": "Result 2"},
            {"title": "Result 3"},
            {"title": "Result 4"},
        ]

        assert (
            search_data_collector_role._needs_followup(sufficient_results, params)
            is False
        )

    def test_llm_guided_followup_search(self, search_data_collector_role):
        """Test LLM-guided follow-up search when needed."""
        results = [{"title": "Insufficient result"}]
        params = {"query": "USS Monitor", "min_results": 3}

        # Mock LLM to suggest follow-up
        mock_agent = Mock()
        mock_agent.return_value = json.dumps(
            {
                "query": "USS Monitor Civil War ironclad",
                "sources": ["academic"],
                "num_results": 3,
            }
        )
        search_data_collector_role.llm_factory.create_agent.return_value = mock_agent

        followup_params = search_data_collector_role._llm_determine_followup(
            results, params
        )

        assert followup_params is not None
        assert followup_params["query"] == "USS Monitor Civil War ironclad"
        assert "academic" in followup_params["sources"]

    def test_error_handling_in_execution(self, search_data_collector_role):
        """Test error handling during search execution."""
        # Mock search tools to raise exception
        search_data_collector_role.search_tools.search_source.side_effect = Exception(
            "Search API failed"
        )

        instruction = "Search for test data"
        result = search_data_collector_role.execute(instruction)

        # Should return empty results but not crash (graceful error handling)
        # The role continues with other sources even if some fail
        assert "search_results" in result
        assert "metadata" in result
        assert (
            result["metadata"]["total_results"] == 0
        )  # No results due to all sources failing
        assert result["execution_type"] == "programmatic_data_collection"

    def test_metrics_tracking(self, search_data_collector_role):
        """Test execution metrics tracking."""
        initial_count = search_data_collector_role.execution_count
        initial_time = search_data_collector_role.total_execution_time

        # Execute task
        search_data_collector_role.execute("Test search instruction")

        # Should have updated metrics
        assert search_data_collector_role.execution_count == initial_count + 1
        assert search_data_collector_role.total_execution_time > initial_time

        # Get metrics
        metrics = search_data_collector_role.get_metrics()
        assert metrics["name"] == "search_data_collector"
        assert metrics["execution_count"] > 0
        assert metrics["average_execution_time"] > 0

    def test_structured_data_output_format(self, search_data_collector_role):
        """Test that output follows structured data format."""
        instruction = "Search for Python programming tutorials"
        result = search_data_collector_role.execute(instruction)

        # Should follow standardized format
        required_fields = ["search_results", "metadata", "execution_type"]
        for field in required_fields:
            assert field in result

        # Search results should be list of dicts
        assert isinstance(result["search_results"], list)
        if result["search_results"]:
            assert isinstance(result["search_results"][0], dict)

        # Metadata should contain required fields
        metadata = result["metadata"]
        required_metadata = [
            "query",
            "sources_searched",
            "total_results",
            "search_timestamp",
            "llm_calls",
        ]
        for field in required_metadata:
            assert field in metadata

    def test_no_analysis_in_output(self, search_data_collector_role):
        """Test that role performs NO analysis of search results."""
        instruction = "Search for machine learning algorithms"
        result = search_data_collector_role.execute(instruction)

        # Should NOT contain analysis fields
        analysis_fields = [
            "summary",
            "analysis",
            "insights",
            "recommendations",
            "conclusions",
        ]
        for field in analysis_fields:
            assert field not in result

        # Search results should be raw data
        for search_result in result["search_results"]:
            # Should contain raw fields only
            raw_fields = ["title", "url", "snippet"]
            for field in raw_fields:
                assert field in search_result or len(search_result) == 0

            # Should NOT contain analysis fields
            for field in analysis_fields:
                assert field not in search_result

    def test_multiple_source_integration(self, search_data_collector_role):
        """Test integration of results from multiple sources."""

        # Configure different results per source
        def mock_search_source(query, source, num_results):
            if source == "web":
                return [{"title": f"Web result for {query}", "source": "web"}]
            elif source == "wikipedia":
                return [
                    {"title": f"Wikipedia result for {query}", "source": "wikipedia"}
                ]
            else:
                return [{"title": f"Other result for {query}", "source": source}]

        search_data_collector_role.search_tools.search_source.side_effect = (
            mock_search_source
        )

        instruction = "Search for artificial intelligence"
        result = search_data_collector_role.execute(instruction)

        # Should have results from multiple sources
        search_results = result["search_results"]
        sources_found = {r.get("source") for r in search_results}

        assert len(sources_found) >= 2  # Should have multiple sources
        assert "web" in sources_found or "wikipedia" in sources_found

    def test_performance_optimization_minimal_llm_calls(
        self, search_data_collector_role
    ):
        """Test that role minimizes LLM calls for performance."""
        instruction = "Search for quantum computing research"

        # Execute task
        result = search_data_collector_role.execute(instruction)

        # Should have minimal LLM calls
        llm_calls = result["metadata"]["llm_calls"]
        assert llm_calls <= 2  # Maximum: 1 for parsing + 1 for follow-up

        # Most common case should be 1 LLM call (just parsing)
        if not search_data_collector_role._needs_followup(
            result["search_results"], {"min_results": 1}
        ):
            assert llm_calls == 1
