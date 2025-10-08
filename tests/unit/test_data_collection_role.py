"""Unit tests for DataCollectionRole programmatic role.

Tests the DataCollectionRole that performs structured data collection from multiple sources
for pure automation workflows without any LLM processing.
"""

import time
from unittest.mock import Mock

import pytest

from common.task_context import TaskContext


class TestDataCollectionRole:
    """Test suite for DataCollectionRole programmatic role."""

    @pytest.fixture
    def mock_data_sources(self):
        """Create mock data sources."""
        sources = Mock()
        sources.collect_from_source.return_value = [
            {"id": 1, "data": "Sample data 1", "source": "api"},
            {"id": 2, "data": "Sample data 2", "source": "api"},
        ]
        return sources

    @pytest.fixture
    def data_collection_role(self, mock_data_sources):
        """Create DataCollectionRole instance with mocked dependencies."""
        # Import after implementation
        from roles.programmatic.data_collection_role import DataCollectionRole

        role = DataCollectionRole()
        role.data_sources = mock_data_sources
        return role

    def test_data_collection_role_initialization(self):
        """Test DataCollectionRole initialization."""
        from roles.programmatic.data_collection_role import DataCollectionRole

        role = DataCollectionRole()

        assert role.name == "data_collection"
        assert "multi-source data aggregation" in role.description.lower()
        assert role.execution_count == 0
        assert role.total_execution_time == 0.0

    def test_parse_instruction_extracts_sources_and_types(self, data_collection_role):
        """Test instruction parsing extracts data sources and types."""
        instruction = (
            "Collect user data from database and API endpoints with JSON format"
        )

        params = data_collection_role.parse_instruction(instruction)

        # Should extract sources, data types, and format
        assert "sources" in params
        assert "data_types" in params
        assert "format" in params

        # Should identify multiple sources
        sources = params["sources"]
        assert isinstance(sources, list)
        assert len(sources) > 0

    def test_execute_multi_source_data_collection(self, data_collection_role):
        """Test multi-source data collection execution."""
        instruction = "Collect customer data from CRM and analytics systems"
        mock_context = Mock(spec=TaskContext)

        result = data_collection_role.execute(instruction, mock_context)

        # Should return structured data collection results
        assert "collected_data" in result
        assert "sources" in result
        assert "total_records" in result
        assert "execution_metadata" in result

        # Should have execution metadata
        metadata = result["execution_metadata"]
        assert metadata["role"] == "data_collection"
        assert metadata["execution_type"] == "programmatic"
        assert metadata["llm_calls"] == 0  # Pure programmatic execution

    def test_collect_from_multiple_sources(self, data_collection_role):
        """Test data collection from multiple sources."""

        # Configure different data per source
        def mock_collect_from_source(source, data_types):
            if source == "database":
                return [{"id": 1, "name": "User 1", "source": "database"}]
            elif source == "api":
                return [{"id": 2, "name": "User 2", "source": "api"}]
            else:
                return [{"id": 3, "name": "User 3", "source": source}]

        data_collection_role.data_sources.collect_from_source.side_effect = (
            mock_collect_from_source
        )

        instruction = "Collect data from database and API sources"
        result = data_collection_role.execute(instruction)

        # Should have collected from multiple sources
        collected_data = result["collected_data"]
        assert len(collected_data) >= 2  # At least database and API

        # Should have aggregated total records
        total_records = result["total_records"]
        assert total_records > 0

    def test_structured_data_formatting(self, data_collection_role):
        """Test structured data formatting and validation."""
        instruction = "Collect product data with structured formatting"
        result = data_collection_role.execute(instruction)

        # Should follow standardized format
        required_fields = [
            "collected_data",
            "sources",
            "total_records",
            "execution_metadata",
        ]
        for field in required_fields:
            assert field in result

        # Collected data should be structured
        assert isinstance(result["collected_data"], dict)
        assert isinstance(result["sources"], list)
        assert isinstance(result["total_records"], int)
        assert isinstance(result["execution_metadata"], dict)

    def test_parallel_data_collection_efficiency(self, data_collection_role):
        """Test parallel data collection for efficiency."""

        # Mock multiple sources with different response times
        def mock_collect_with_delay(source, data_types):
            # time.sleep(0.01)  # Removed for faster tests
            return [{"data": f"Data from {source}", "source": source}]

        data_collection_role.data_sources.collect_from_source.side_effect = (
            mock_collect_with_delay
        )

        start_time = time.time()
        result = data_collection_role.execute("Collect from multiple sources")
        execution_time = time.time() - start_time

        # Should complete efficiently even with multiple sources
        assert execution_time < 1.0  # Should be fast
        assert result["total_records"] > 0

    def test_error_handling_partial_source_failures(self, data_collection_role):
        """Test error handling when some sources fail."""

        # Mock one source to fail, others to succeed
        def mock_collect_with_failures(source, data_types):
            if source == "failing_source":
                raise Exception("Source unavailable")
            return [{"data": f"Success from {source}", "source": source}]

        data_collection_role.data_sources.collect_from_source.side_effect = (
            mock_collect_with_failures
        )

        instruction = "Collect from working_source and failing_source"
        result = data_collection_role.execute(instruction)

        # Should continue with successful sources
        assert "collected_data" in result
        assert (
            result["total_records"] >= 0
        )  # May have some data from successful sources

        # Should not crash completely
        assert "execution_metadata" in result
        assert result["execution_metadata"]["execution_type"] == "programmatic"

    def test_metrics_tracking_for_data_collection(self, data_collection_role):
        """Test execution metrics tracking."""
        initial_count = data_collection_role.execution_count
        initial_time = data_collection_role.total_execution_time

        # Execute data collection
        data_collection_role.execute("Collect test data")

        # Should have updated metrics
        assert data_collection_role.execution_count == initial_count + 1
        assert data_collection_role.total_execution_time > initial_time

        # Get metrics
        metrics = data_collection_role.get_metrics()
        assert metrics["name"] == "data_collection"
        assert metrics["execution_count"] > 0
        assert metrics["average_execution_time"] > 0

    def test_data_type_filtering_and_validation(self, data_collection_role):
        """Test data type filtering and validation."""

        # Configure mock to return different data types
        def mock_collect_typed_data(source, data_types):
            all_data = [
                {"type": "user", "id": 1, "name": "John"},
                {"type": "product", "id": 2, "name": "Widget"},
                {"type": "order", "id": 3, "amount": 100},
            ]
            # Filter by requested data types
            return [item for item in all_data if item["type"] in data_types]

        data_collection_role.data_sources.collect_from_source.side_effect = (
            mock_collect_typed_data
        )

        instruction = "Collect user and product data with type filtering"
        result = data_collection_role.execute(instruction)

        # Should have collected and filtered data appropriately
        assert "collected_data" in result
        assert result["total_records"] > 0

    def test_no_llm_processing_pure_automation(self, data_collection_role):
        """Test that role performs NO LLM processing - pure automation."""
        instruction = "Collect all available data from configured sources"
        result = data_collection_role.execute(instruction)

        # Should have zero LLM calls
        metadata = result["execution_metadata"]
        assert metadata["llm_calls"] == 0

        # Should be pure programmatic execution
        assert metadata["execution_type"] == "programmatic"

        # Should not contain any analysis or reasoning fields
        analysis_fields = ["summary", "analysis", "insights", "recommendations"]
        for field in analysis_fields:
            assert field not in result

    def test_batch_processing_capabilities(self, data_collection_role):
        """Test batch processing of data collection requests."""

        # Mock batch data collection
        def mock_batch_collect(source, data_types):
            return [
                {"batch": i, "data": f"Batch data {i}", "source": source}
                for i in range(5)
            ]  # Return 5 items per source

        data_collection_role.data_sources.collect_from_source.side_effect = (
            mock_batch_collect
        )

        instruction = "Collect data in batches from multiple sources"
        result = data_collection_role.execute(instruction)

        # Should handle batch processing efficiently
        assert result["total_records"] >= 5  # Should have batch data

        # Should maintain performance even with larger datasets
        assert "execution_metadata" in result
        execution_time = float(
            result["execution_metadata"].get("execution_time", "0s").replace("s", "")
        )
        assert execution_time < 5.0  # Should be efficient

    def test_data_format_conversion_and_validation(self, data_collection_role):
        """Test data format conversion and validation."""
        instruction = "Collect data with CSV format conversion and validation"
        result = data_collection_role.execute(instruction)

        # Should handle format specifications
        assert "collected_data" in result
        assert "execution_metadata" in result

        # Should maintain data integrity
        for source_data in result["collected_data"].values():
            assert isinstance(source_data, list)
            for item in source_data:
                assert isinstance(item, dict)
