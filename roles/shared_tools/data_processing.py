"""
Data Processing Shared Tools

Common data analysis and formatting functionality that can be used across multiple roles.
"""

import json
import logging
import statistics
from datetime import datetime
from typing import Any, Dict, List, Union

from strands import tool

logger = logging.getLogger(__name__)


@tool
def analyze_data(data: Union[list, dict], analysis_type: str = "summary") -> dict:
    """
    Perform basic data analysis on structured data.

    Args:
        data: Data to analyze (list or dict)
        analysis_type: Type of analysis ("summary", "statistics", "structure")

    Returns:
        Dict containing analysis results
    """
    try:
        if analysis_type == "summary":
            return _analyze_summary(data)
        elif analysis_type == "statistics":
            return _analyze_statistics(data)
        elif analysis_type == "structure":
            return _analyze_structure(data)
        else:
            return {
                "success": False,
                "error": f"Unknown analysis type: {analysis_type}",
                "supported_types": ["summary", "statistics", "structure"],
            }

    except Exception as e:
        logger.error(f"Data analysis failed: {e}")
        return {"success": False, "error": str(e), "analysis_type": analysis_type}


def _analyze_summary(data: Union[list, dict]) -> dict:
    """Generate summary analysis of data."""
    if isinstance(data, list):
        return {
            "success": True,
            "data_type": "list",
            "total_items": len(data),
            "item_types": list({type(item).__name__ for item in data}),
            "sample_items": data[:3] if data else [],
            "is_empty": len(data) == 0,
        }
    elif isinstance(data, dict):
        return {
            "success": True,
            "data_type": "dict",
            "total_keys": len(data),
            "keys": list(data.keys()),
            "value_types": list({type(value).__name__ for value in data.values()}),
            "sample_entries": dict(list(data.items())[:3]) if data else {},
            "is_empty": len(data) == 0,
        }
    else:
        return {
            "success": True,
            "data_type": type(data).__name__,
            "value": str(data),
            "length": len(str(data)),
        }


def _analyze_statistics(data: Union[list, dict]) -> dict:
    """Generate statistical analysis of numerical data."""
    try:
        # Extract numerical values
        numbers = []
        if isinstance(data, list):
            numbers = [x for x in data if isinstance(x, (int, float))]
        elif isinstance(data, dict):
            numbers = [v for v in data.values() if isinstance(v, (int, float))]

        if not numbers:
            return {
                "success": False,
                "error": "No numerical data found for statistical analysis",
            }

        return {
            "success": True,
            "count": len(numbers),
            "mean": statistics.mean(numbers),
            "median": statistics.median(numbers),
            "mode": (
                statistics.mode(numbers) if len(set(numbers)) < len(numbers) else None
            ),
            "std_dev": statistics.stdev(numbers) if len(numbers) > 1 else 0,
            "min": min(numbers),
            "max": max(numbers),
            "range": max(numbers) - min(numbers),
        }

    except Exception as e:
        return {"success": False, "error": f"Statistical analysis failed: {e}"}


def _analyze_structure(data: Union[list, dict]) -> dict:
    """Analyze the structure of complex data."""

    def get_structure(obj, max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return f"<max_depth_reached: {type(obj).__name__}>"

        if isinstance(obj, dict):
            return {
                k: get_structure(v, max_depth, current_depth + 1)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            if obj:
                return [get_structure(obj[0], max_depth, current_depth + 1)]
            else:
                return []
        else:
            return type(obj).__name__

    return {
        "success": True,
        "structure": get_structure(data),
        "data_type": type(data).__name__,
        "analysis_timestamp": datetime.now().isoformat(),
    }


@tool
def format_output(data: Any, format_type: str = "json", pretty: bool = True) -> dict:
    """
    Format data for output in various formats.

    Args:
        data: Data to format
        format_type: Output format ("json", "yaml", "table", "list")
        pretty: Whether to use pretty formatting

    Returns:
        Dict containing formatted output
    """
    try:
        if format_type == "json":
            formatted = json.dumps(data, indent=2 if pretty else None, default=str)
        elif format_type == "yaml":
            import yaml

            formatted = yaml.dump(
                data, default_flow_style=not pretty, indent=2 if pretty else None
            )
        elif format_type == "table":
            formatted = _format_as_table(data)
        elif format_type == "list":
            formatted = _format_as_list(data)
        else:
            return {
                "success": False,
                "error": f"Unknown format type: {format_type}",
                "supported_formats": ["json", "yaml", "table", "list"],
            }

        return {
            "success": True,
            "formatted_output": formatted,
            "format_type": format_type,
            "character_count": len(formatted),
        }

    except Exception as e:
        logger.error(f"Output formatting failed: {e}")
        return {"success": False, "error": str(e), "format_type": format_type}


def _format_as_table(data: Any) -> str:
    """Format data as a simple table."""
    if isinstance(data, list) and data and isinstance(data[0], dict):
        # List of dicts - create table
        headers = list(data[0].keys())
        rows = []

        # Header row
        header_row = " | ".join(str(h) for h in headers)
        separator = " | ".join("-" * len(str(h)) for h in headers)
        rows.append(header_row)
        rows.append(separator)

        # Data rows
        for item in data[:10]:  # Limit to 10 rows
            row = " | ".join(str(item.get(h, "")) for h in headers)
            rows.append(row)

        return "\n".join(rows)
    else:
        return str(data)


def _format_as_list(data: Any) -> str:
    """Format data as a bulleted list."""
    if isinstance(data, list):
        return "\n".join(f"• {item}" for item in data)
    elif isinstance(data, dict):
        return "\n".join(f"• {k}: {v}" for k, v in data.items())
    else:
        return f"• {data}"


@tool
def extract_key_information(text: str, info_type: str = "entities") -> dict:
    """
    Extract key information from text.

    Args:
        text: Text to analyze
        info_type: Type of information to extract ("entities", "keywords", "summary")

    Returns:
        Dict containing extracted information
    """
    try:
        if info_type == "entities":
            # Simple entity extraction (would use NLP library in production)
            words = text.split()
            entities = [word for word in words if word[0].isupper() and len(word) > 2]
            return {
                "success": True,
                "entities": list(set(entities)),
                "entity_count": len(set(entities)),
                "info_type": info_type,
            }
        elif info_type == "keywords":
            # Simple keyword extraction
            words = text.lower().split()
            keywords = [word for word in words if len(word) > 4]
            return {
                "success": True,
                "keywords": list(set(keywords))[:10],  # Top 10 keywords
                "keyword_count": len(set(keywords)),
                "info_type": info_type,
            }
        elif info_type == "summary":
            # Simple summary (first and last sentences)
            sentences = text.split(".")
            summary = (
                f"{sentences[0]}. ... {sentences[-1]}" if len(sentences) > 2 else text
            )
            return {
                "success": True,
                "summary": summary.strip(),
                "original_length": len(text),
                "summary_length": len(summary),
                "info_type": info_type,
            }
        else:
            return {
                "success": False,
                "error": f"Unknown info type: {info_type}",
                "supported_types": ["entities", "keywords", "summary"],
            }

    except Exception as e:
        logger.error(f"Information extraction failed: {e}")
        return {"success": False, "error": str(e), "info_type": info_type}


@tool
def perform_comparative_analysis(
    data1: Any, data2: Any, comparison_type: str = "basic"
) -> dict:
    """
    Perform comparative analysis between two datasets or pieces of information.

    Args:
        data1: First dataset or information to compare
        data2: Second dataset or information to compare
        comparison_type: Type of comparison ("basic", "statistical", "structural")

    Returns:
        Dict containing comparative analysis results
    """
    try:
        if comparison_type == "basic":
            return _basic_comparison(data1, data2)
        elif comparison_type == "statistical":
            return _statistical_comparison(data1, data2)
        elif comparison_type == "structural":
            return _structural_comparison(data1, data2)
        else:
            return {
                "success": False,
                "error": f"Unknown comparison type: {comparison_type}",
                "supported_types": ["basic", "statistical", "structural"],
            }

    except Exception as e:
        logger.error(f"Comparative analysis failed: {e}")
        return {"success": False, "error": str(e), "comparison_type": comparison_type}


def _basic_comparison(data1: Any, data2: Any) -> dict:
    """Perform basic comparison between two pieces of data."""
    return {
        "success": True,
        "data1_type": type(data1).__name__,
        "data2_type": type(data2).__name__,
        "types_match": type(data1) is type(data2),
        "data1_size": len(str(data1)),
        "data2_size": len(str(data2)),
        "size_difference": len(str(data1)) - len(str(data2)),
        "are_equal": data1 == data2,
        "comparison_timestamp": datetime.now().isoformat(),
    }


def _statistical_comparison(data1: Any, data2: Any) -> dict:
    """Perform statistical comparison between numerical datasets."""
    try:
        # Extract numerical values
        nums1 = []
        nums2 = []

        if isinstance(data1, list):
            nums1 = [x for x in data1 if isinstance(x, (int, float))]
        elif isinstance(data1, (int, float)):
            nums1 = [data1]

        if isinstance(data2, list):
            nums2 = [x for x in data2 if isinstance(x, (int, float))]
        elif isinstance(data2, (int, float)):
            nums2 = [data2]

        if not nums1 or not nums2:
            return {
                "success": False,
                "error": "Insufficient numerical data for statistical comparison",
            }

        return {
            "success": True,
            "dataset1": {
                "count": len(nums1),
                "mean": statistics.mean(nums1),
                "median": statistics.median(nums1),
                "std_dev": statistics.stdev(nums1) if len(nums1) > 1 else 0,
            },
            "dataset2": {
                "count": len(nums2),
                "mean": statistics.mean(nums2),
                "median": statistics.median(nums2),
                "std_dev": statistics.stdev(nums2) if len(nums2) > 1 else 0,
            },
            "comparison": {
                "mean_difference": statistics.mean(nums1) - statistics.mean(nums2),
                "median_difference": statistics.median(nums1)
                - statistics.median(nums2),
                "size_difference": len(nums1) - len(nums2),
            },
        }

    except Exception as e:
        return {"success": False, "error": f"Statistical comparison failed: {e}"}


def _structural_comparison(data1: Any, data2: Any) -> dict:
    """Compare the structure of two complex data objects."""

    def get_structure_signature(obj):
        if isinstance(obj, dict):
            return {k: get_structure_signature(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [get_structure_signature(obj[0])] if obj else []
        else:
            return type(obj).__name__

    struct1 = get_structure_signature(data1)
    struct2 = get_structure_signature(data2)

    return {
        "success": True,
        "structure1": struct1,
        "structure2": struct2,
        "structures_match": struct1 == struct2,
        "data1_complexity": _calculate_complexity(data1),
        "data2_complexity": _calculate_complexity(data2),
    }


def _calculate_complexity(data: Any) -> int:
    """Calculate complexity score for data structure."""
    if isinstance(data, dict):
        return 1 + sum(_calculate_complexity(v) for v in data.values())
    elif isinstance(data, list):
        return 1 + sum(
            _calculate_complexity(item) for item in data[:5]
        )  # Sample first 5
    else:
        return 1


@tool
def generate_insights(data: Any, focus_area: str = "patterns") -> dict:
    """
    Generate insights and observations from data.

    Args:
        data: Data to analyze for insights
        focus_area: Area to focus on ("patterns", "trends", "anomalies", "relationships")

    Returns:
        Dict containing generated insights
    """
    try:
        insights = []

        if focus_area == "patterns":
            insights = _identify_patterns(data)
        elif focus_area == "trends":
            insights = _identify_trends(data)
        elif focus_area == "anomalies":
            insights = _identify_anomalies(data)
        elif focus_area == "relationships":
            insights = _identify_relationships(data)
        else:
            return {
                "success": False,
                "error": f"Unknown focus area: {focus_area}",
                "supported_areas": ["patterns", "trends", "anomalies", "relationships"],
            }

        return {
            "success": True,
            "focus_area": focus_area,
            "insights": insights,
            "insight_count": len(insights),
            "analysis_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Insight generation failed: {e}")
        return {"success": False, "error": str(e), "focus_area": focus_area}


def _identify_patterns(data: Any) -> list[str]:
    """Identify patterns in data."""
    patterns = []

    if isinstance(data, list):
        if len(data) > 1:
            patterns.append(f"Dataset contains {len(data)} items")
            if all(isinstance(x, type(data[0])) for x in data):
                patterns.append(f"All items are of type {type(data[0]).__name__}")
            if isinstance(data[0], (int, float)):
                if all(x > 0 for x in data):
                    patterns.append("All values are positive")
                elif all(x < 0 for x in data):
                    patterns.append("All values are negative")

    elif isinstance(data, dict):
        patterns.append(f"Dictionary with {len(data)} keys")
        if data:
            value_types = [type(v).__name__ for v in data.values()]
            if len(set(value_types)) == 1:
                patterns.append(f"All values are of type {value_types[0]}")

    return patterns or ["No clear patterns identified"]


def _identify_trends(data: Any) -> list[str]:
    """Identify trends in data."""
    trends = []

    if isinstance(data, list) and len(data) > 2:
        if all(isinstance(x, (int, float)) for x in data):
            if all(data[i] <= data[i + 1] for i in range(len(data) - 1)):
                trends.append("Data shows consistent upward trend")
            elif all(data[i] >= data[i + 1] for i in range(len(data) - 1)):
                trends.append("Data shows consistent downward trend")
            else:
                trends.append("Data shows mixed trend pattern")

    return trends or ["No clear trends identified"]


def _identify_anomalies(data: Any) -> list[str]:
    """Identify anomalies or outliers in data."""
    anomalies = []

    if isinstance(data, list) and len(data) > 3:
        if all(isinstance(x, (int, float)) for x in data):
            mean_val = statistics.mean(data)
            std_dev = statistics.stdev(data) if len(data) > 1 else 0

            if std_dev > 0:
                outliers = [x for x in data if abs(x - mean_val) > 2 * std_dev]
                if outliers:
                    anomalies.append(f"Found {len(outliers)} statistical outliers")

    return anomalies or ["No significant anomalies detected"]


def _identify_relationships(data: Any) -> list[str]:
    """Identify relationships within data."""
    relationships = []

    if isinstance(data, dict):
        # Look for key-value relationships
        numeric_values = [v for v in data.values() if isinstance(v, (int, float))]
        if len(numeric_values) > 1:
            relationships.append(
                f"Contains {len(numeric_values)} numerical relationships"
            )

        string_values = [v for v in data.values() if isinstance(v, str)]
        if len(string_values) > 1:
            relationships.append(f"Contains {len(string_values)} textual relationships")

    return relationships or ["No clear relationships identified"]
