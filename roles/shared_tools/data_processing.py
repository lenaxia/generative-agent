"""
Data Processing Shared Tools

Common data analysis and formatting functionality that can be used across multiple roles.
"""

from strands import tool
from typing import Dict, List, Any, Optional, Union
import json
import logging
import statistics
from datetime import datetime

logger = logging.getLogger(__name__)


@tool
def analyze_data(data: Union[List, Dict], analysis_type: str = "summary") -> Dict:
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
                "supported_types": ["summary", "statistics", "structure"]
            }
            
    except Exception as e:
        logger.error(f"Data analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "analysis_type": analysis_type
        }


def _analyze_summary(data: Union[List, Dict]) -> Dict:
    """Generate summary analysis of data."""
    if isinstance(data, list):
        return {
            "success": True,
            "data_type": "list",
            "total_items": len(data),
            "item_types": list(set(type(item).__name__ for item in data)),
            "sample_items": data[:3] if data else [],
            "is_empty": len(data) == 0
        }
    elif isinstance(data, dict):
        return {
            "success": True,
            "data_type": "dict",
            "total_keys": len(data),
            "keys": list(data.keys()),
            "value_types": list(set(type(value).__name__ for value in data.values())),
            "sample_entries": dict(list(data.items())[:3]) if data else {},
            "is_empty": len(data) == 0
        }
    else:
        return {
            "success": True,
            "data_type": type(data).__name__,
            "value": str(data),
            "length": len(str(data))
        }


def _analyze_statistics(data: Union[List, Dict]) -> Dict:
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
                "error": "No numerical data found for statistical analysis"
            }
        
        return {
            "success": True,
            "count": len(numbers),
            "mean": statistics.mean(numbers),
            "median": statistics.median(numbers),
            "mode": statistics.mode(numbers) if len(set(numbers)) < len(numbers) else None,
            "std_dev": statistics.stdev(numbers) if len(numbers) > 1 else 0,
            "min": min(numbers),
            "max": max(numbers),
            "range": max(numbers) - min(numbers)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Statistical analysis failed: {e}"
        }


def _analyze_structure(data: Union[List, Dict]) -> Dict:
    """Analyze the structure of complex data."""
    def get_structure(obj, max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return f"<max_depth_reached: {type(obj).__name__}>"
        
        if isinstance(obj, dict):
            return {k: get_structure(v, max_depth, current_depth + 1) for k, v in obj.items()}
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
        "analysis_timestamp": datetime.now().isoformat()
    }


@tool
def format_output(data: Any, format_type: str = "json", pretty: bool = True) -> Dict:
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
            formatted = yaml.dump(data, default_flow_style=not pretty, indent=2 if pretty else None)
        elif format_type == "table":
            formatted = _format_as_table(data)
        elif format_type == "list":
            formatted = _format_as_list(data)
        else:
            return {
                "success": False,
                "error": f"Unknown format type: {format_type}",
                "supported_formats": ["json", "yaml", "table", "list"]
            }
        
        return {
            "success": True,
            "formatted_output": formatted,
            "format_type": format_type,
            "character_count": len(formatted)
        }
        
    except Exception as e:
        logger.error(f"Output formatting failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "format_type": format_type
        }


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
def extract_key_information(text: str, info_type: str = "entities") -> Dict:
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
                "info_type": info_type
            }
        elif info_type == "keywords":
            # Simple keyword extraction
            words = text.lower().split()
            keywords = [word for word in words if len(word) > 4]
            return {
                "success": True,
                "keywords": list(set(keywords))[:10],  # Top 10 keywords
                "keyword_count": len(set(keywords)),
                "info_type": info_type
            }
        elif info_type == "summary":
            # Simple summary (first and last sentences)
            sentences = text.split('.')
            summary = f"{sentences[0]}. ... {sentences[-1]}" if len(sentences) > 2 else text
            return {
                "success": True,
                "summary": summary.strip(),
                "original_length": len(text),
                "summary_length": len(summary),
                "info_type": info_type
            }
        else:
            return {
                "success": False,
                "error": f"Unknown info type: {info_type}",
                "supported_types": ["entities", "keywords", "summary"]
            }
            
    except Exception as e:
        logger.error(f"Information extraction failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "info_type": info_type
        }