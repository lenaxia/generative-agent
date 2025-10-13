"""Research Analyst Custom Tools

Specialized tools for research and analysis tasks.
"""

import logging
import statistics
from datetime import datetime

from strands import tool

logger = logging.getLogger(__name__)


@tool
def academic_search(
    query: str, databases: list[str] = None, max_results: int = 10
) -> dict:
    """Search academic databases for research papers and scholarly articles.

    Args:
        query: Academic search query
        databases: List of databases to search (e.g., ["pubmed", "arxiv", "scholar"])
        max_results: Maximum number of results to return

    Returns:
        Dict containing academic search results with citations
    """
    try:
        # Default databases if none specified
        if databases is None:
            databases = ["scholar", "pubmed", "arxiv"]

        # Mock academic search results (would integrate with real APIs)
        results = []
        for i in range(min(max_results, 5)):
            results.append(
                {
                    "title": f"Academic Paper {i+1}: {query}",
                    "authors": [f"Author {i+1}", f"Co-Author {i+1}"],
                    "journal": f"Journal of {query.title()} Studies",
                    "year": 2024 - i,
                    "doi": f"10.1000/journal.{i+1}",
                    "abstract": f"This paper investigates {query} through comprehensive analysis...",
                    "citation_count": 150 - (i * 20),
                    "url": f"https://scholar.example.com/paper-{i+1}",
                    "database": databases[i % len(databases)],
                }
            )

        return {
            "query": query,
            "databases_searched": databases,
            "results": results,
            "total_results": len(results),
            "search_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Academic search failed: {e}")
        return {"query": query, "results": [], "error": str(e)}


@tool
def citation_formatter(papers: list[dict], style: str = "APA") -> dict:
    """Format research citations in various academic styles.

    Args:
        papers: List of paper dictionaries with citation information
        style: Citation style ("APA", "MLA", "Chicago", "IEEE")

    Returns:
        Dict containing formatted citations
    """
    try:
        formatted_citations = []

        for paper in papers:
            if style.upper() == "APA":
                citation = _format_apa_citation(paper)
            elif style.upper() == "MLA":
                citation = _format_mla_citation(paper)
            elif style.upper() == "CHICAGO":
                citation = _format_chicago_citation(paper)
            elif style.upper() == "IEEE":
                citation = _format_ieee_citation(paper)
            else:
                citation = f"Unknown style: {style}"

            formatted_citations.append(citation)

        return {
            "style": style,
            "citation_count": len(formatted_citations),
            "citations": formatted_citations,
            "bibliography": "\n\n".join(formatted_citations),
        }

    except Exception as e:
        logger.error(f"Citation formatting failed: {e}")
        return {"style": style, "citations": [], "error": str(e)}


def _format_apa_citation(paper: dict) -> str:
    """Format citation in APA style."""
    authors = paper.get("authors", ["Unknown Author"])
    author_str = ", ".join(authors) if len(authors) <= 2 else f"{authors[0]} et al."

    title = paper.get("title", "Untitled")
    journal = paper.get("journal", "Unknown Journal")
    year = paper.get("year", "n.d.")

    return f"{author_str} ({year}). {title}. {journal}."


def _format_mla_citation(paper: dict) -> str:
    """Format citation in MLA style."""
    authors = paper.get("authors", ["Unknown Author"])
    author_str = authors[0] if authors else "Unknown Author"

    title = paper.get("title", "Untitled")
    journal = paper.get("journal", "Unknown Journal")
    year = paper.get("year", "n.d.")

    return f'{author_str}. "{title}." {journal}, {year}.'


def _format_chicago_citation(paper: dict) -> str:
    """Format citation in Chicago style."""
    authors = paper.get("authors", ["Unknown Author"])
    author_str = authors[0] if authors else "Unknown Author"

    title = paper.get("title", "Untitled")
    journal = paper.get("journal", "Unknown Journal")
    year = paper.get("year", "n.d.")

    return f'{author_str}. "{title}." {journal} ({year}).'


def _format_ieee_citation(paper: dict) -> str:
    """Format citation in IEEE style."""
    authors = paper.get("authors", ["Unknown Author"])
    author_str = ", ".join(authors) if len(authors) <= 3 else f"{authors[0]} et al."

    title = paper.get("title", "Untitled")
    journal = paper.get("journal", "Unknown Journal")
    year = paper.get("year", "n.d.")

    return f'{author_str}, "{title}," {journal}, {year}.'


@tool
def statistical_analyzer(data: list[float], analysis_type: str = "descriptive") -> dict:
    """Perform statistical analysis on numerical data.

    Args:
        data: List of numerical values
        analysis_type: Type of analysis ("descriptive", "correlation", "trend")

    Returns:
        Dict containing statistical analysis results
    """
    try:
        if not data or not all(isinstance(x, (int, float)) for x in data):
            return {
                "success": False,
                "error": "Invalid data: must be a list of numbers",
                "analysis_type": analysis_type,
            }

        if analysis_type == "descriptive":
            return _descriptive_statistics(data)
        elif analysis_type == "correlation":
            return _correlation_analysis(data)
        elif analysis_type == "trend":
            return _trend_analysis(data)
        else:
            return {
                "success": False,
                "error": f"Unknown analysis type: {analysis_type}",
                "supported_types": ["descriptive", "correlation", "trend"],
            }

    except Exception as e:
        logger.error(f"Statistical analysis failed: {e}")
        return {"success": False, "error": str(e), "analysis_type": analysis_type}


def _descriptive_statistics(data: list[float]) -> dict:
    """Calculate descriptive statistics."""
    import statistics

    return {
        "success": True,
        "analysis_type": "descriptive",
        "count": len(data),
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "mode": statistics.mode(data) if len(set(data)) < len(data) else None,
        "std_dev": statistics.stdev(data) if len(data) > 1 else 0,
        "variance": statistics.variance(data) if len(data) > 1 else 0,
        "min": min(data),
        "max": max(data),
        "range": max(data) - min(data),
        "quartiles": {
            "q1": statistics.quantiles(data, n=4)[0] if len(data) >= 4 else None,
            "q2": statistics.median(data),
            "q3": statistics.quantiles(data, n=4)[2] if len(data) >= 4 else None,
        },
    }


def _correlation_analysis(data: list[float]) -> dict:
    """Analyze correlation patterns in data."""
    # Simple autocorrelation analysis
    if len(data) < 2:
        return {
            "success": False,
            "error": "Need at least 2 data points for correlation analysis",
        }

    # Calculate simple trend correlation
    x_values = list(range(len(data)))
    correlation = _calculate_correlation(x_values, data)

    return {
        "success": True,
        "analysis_type": "correlation",
        "time_correlation": correlation,
        "trend_direction": (
            "positive"
            if correlation > 0.1
            else "negative"
            if correlation < -0.1
            else "neutral"
        ),
        "correlation_strength": abs(correlation),
    }


def _trend_analysis(data: list[float]) -> dict:
    """Analyze trends in time series data."""
    if len(data) < 3:
        return {
            "success": False,
            "error": "Need at least 3 data points for trend analysis",
        }

    # Simple trend detection
    increases = sum(1 for i in range(1, len(data)) if data[i] > data[i - 1])
    decreases = sum(1 for i in range(1, len(data)) if data[i] < data[i - 1])

    trend = (
        "increasing"
        if increases > decreases
        else "decreasing"
        if decreases > increases
        else "stable"
    )

    return {
        "success": True,
        "analysis_type": "trend",
        "trend_direction": trend,
        "increases": increases,
        "decreases": decreases,
        "stability_ratio": abs(increases - decreases) / (len(data) - 1),
        "volatility": (
            statistics.stdev(data) / statistics.mean(data)
            if statistics.mean(data) != 0
            else 0
        ),
    }


def _calculate_correlation(x: list[float], y: list[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    import statistics

    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
    sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
    sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))

    denominator = (sum_sq_x * sum_sq_y) ** 0.5

    return numerator / denominator if denominator != 0 else 0.0


@tool
def research_report_generator(
    findings: list[dict], title: str, include_methodology: bool = True
) -> dict:
    """Generate a structured research report from findings.

    Args:
        findings: List of research findings and data
        title: Title for the research report
        include_methodology: Whether to include methodology section

    Returns:
        Dict containing structured research report
    """
    try:
        report_sections = []

        # Title and Executive Summary
        report_sections.append(f"# {title}")
        report_sections.append(
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_sections.append("")

        # Executive Summary
        report_sections.append("## Executive Summary")
        report_sections.append(
            f"This report presents analysis based on {len(findings)} key findings."
        )
        report_sections.append("")

        # Methodology (if requested)
        if include_methodology:
            report_sections.append("## Methodology")
            report_sections.append("- Comprehensive literature review")
            report_sections.append("- Multi-source data analysis")
            report_sections.append("- Statistical analysis where applicable")
            report_sections.append("- Evidence-based synthesis")
            report_sections.append("")

        # Findings
        report_sections.append("## Key Findings")
        for i, finding in enumerate(findings, 1):
            report_sections.append(f"### Finding {i}")
            if isinstance(finding, dict):
                for key, value in finding.items():
                    report_sections.append(f"- **{key.title()}:** {value}")
            else:
                report_sections.append(f"- {finding}")
            report_sections.append("")

        # Conclusions and Recommendations
        report_sections.append("## Conclusions and Recommendations")
        report_sections.append("Based on the analysis conducted:")
        report_sections.append("- Further research is recommended in identified areas")
        report_sections.append("- Key patterns and trends have been identified")
        report_sections.append("- Evidence supports the presented findings")
        report_sections.append("")

        # References placeholder
        report_sections.append("## References")
        report_sections.append("*Citations and references would be listed here*")

        full_report = "\n".join(report_sections)

        return {
            "success": True,
            "title": title,
            "report": full_report,
            "word_count": len(full_report.split()),
            "section_count": len([s for s in report_sections if s.startswith("#")]),
            "findings_count": len(findings),
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {"success": False, "error": str(e), "title": title}
