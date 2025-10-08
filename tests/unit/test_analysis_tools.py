"""Unit tests for analysis tools."""

from roles.shared_tools.data_processing import (
    generate_insights,
    perform_comparative_analysis,
)


class TestAnalysisTools:
    """Test cases for analysis functionality."""

    def test_perform_comparative_analysis_basic(self):
        """Test basic comparative analysis."""
        data1 = [1, 2, 3, 4, 5]
        data2 = [2, 3, 4, 5, 6]

        result = perform_comparative_analysis(data1, data2, "basic")

        assert result["success"] is True
        assert result["data1_type"] == "list"
        assert result["data2_type"] == "list"
        assert result["types_match"] is True
        assert result["are_equal"] is False
        assert (
            result["size_difference"] == 0
        )  # Both have same string length representation

    def test_perform_comparative_analysis_statistical(self):
        """Test statistical comparative analysis."""
        data1 = [1, 2, 3, 4, 5]
        data2 = [3, 4, 5, 6, 7]

        result = perform_comparative_analysis(data1, data2, "statistical")

        assert result["success"] is True
        assert "dataset1" in result
        assert "dataset2" in result
        assert "comparison" in result
        assert result["dataset1"]["count"] == 5
        assert result["dataset2"]["count"] == 5
        assert result["dataset1"]["mean"] == 3.0
        assert result["dataset2"]["mean"] == 5.0
        assert result["comparison"]["mean_difference"] == -2.0

    def test_perform_comparative_analysis_structural(self):
        """Test structural comparative analysis."""
        data1 = {"name": "John", "age": 30, "city": "NYC"}
        data2 = {"name": "Jane", "age": 25, "city": "LA"}

        result = perform_comparative_analysis(data1, data2, "structural")

        assert result["success"] is True
        assert "structure1" in result
        assert "structure2" in result
        assert result["structures_match"] is True  # Same structure
        assert result["data1_complexity"] > 0
        assert result["data2_complexity"] > 0

    def test_perform_comparative_analysis_invalid_type(self):
        """Test comparative analysis with invalid type."""
        result = perform_comparative_analysis([1, 2, 3], [4, 5, 6], "invalid")

        assert result["success"] is False
        assert "error" in result
        assert "Unknown comparison type" in result["error"]
        assert "supported_types" in result

    def test_perform_comparative_analysis_no_numerical_data(self):
        """Test statistical analysis with no numerical data."""
        data1 = ["a", "b", "c"]
        data2 = ["d", "e", "f"]

        result = perform_comparative_analysis(data1, data2, "statistical")

        assert result["success"] is False
        assert "error" in result
        assert "Insufficient numerical data" in result["error"]

    def test_generate_insights_patterns(self):
        """Test insight generation for patterns."""
        data = [1, 2, 3, 4, 5]

        result = generate_insights(data, "patterns")

        assert result["success"] is True
        assert result["focus_area"] == "patterns"
        assert "insights" in result
        assert result["insight_count"] > 0
        assert isinstance(result["insights"], list)

    def test_generate_insights_trends(self):
        """Test insight generation for trends."""
        data = [1, 2, 3, 4, 5]  # Upward trend

        result = generate_insights(data, "trends")

        assert result["success"] is True
        assert result["focus_area"] == "trends"
        assert "insights" in result
        assert any("upward trend" in insight for insight in result["insights"])

    def test_generate_insights_anomalies(self):
        """Test insight generation for anomalies."""
        data = [1, 2, 3, 100, 5]  # 100 is an outlier

        result = generate_insights(data, "anomalies")

        assert result["success"] is True
        assert result["focus_area"] == "anomalies"
        assert "insights" in result

    def test_generate_insights_relationships(self):
        """Test insight generation for relationships."""
        data = {
            "price": 100,
            "quantity": 5,
            "name": "Product A",
            "category": "Electronics",
        }

        result = generate_insights(data, "relationships")

        assert result["success"] is True
        assert result["focus_area"] == "relationships"
        assert "insights" in result
        assert result["insight_count"] > 0

    def test_generate_insights_invalid_focus(self):
        """Test insight generation with invalid focus area."""
        result = generate_insights([1, 2, 3], "invalid")

        assert result["success"] is False
        assert "error" in result
        assert "Unknown focus area" in result["error"]
        assert "supported_areas" in result

    def test_generate_insights_empty_data(self):
        """Test insight generation with empty data."""
        result = generate_insights([], "patterns")

        assert result["success"] is True
        assert result["insights"] == ["No clear patterns identified"]

    def test_comparative_analysis_different_types(self):
        """Test comparative analysis with different data types."""
        data1 = [1, 2, 3]
        data2 = {"a": 1, "b": 2, "c": 3}

        result = perform_comparative_analysis(data1, data2, "basic")

        assert result["success"] is True
        assert result["data1_type"] == "list"
        assert result["data2_type"] == "dict"
        assert result["types_match"] is False
        assert result["are_equal"] is False
