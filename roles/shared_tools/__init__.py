"""Shared Tools Module

Common tools that can be used across multiple roles.
"""

from .data_processing import analyze_data, extract_key_information, format_output
from .file_operations import (
    list_directory_contents,
    read_file_content,
    write_file_content,
)
from .search_pipeline import (
    multi_source_search_pipeline,
    search_and_scrape_pipeline,
    wikipedia_search_pipeline,
)
from .slack_tools import send_slack_message
from .summarizer_tools import summarize_text
from .weather_tools import get_weather
from .web_search import (
    extract_article_content,
    scrape_webpage,
    scrape_with_links,
    search_academic,
    search_news,
    search_with_filters,
    web_search,
)

__all__ = [
    # Web search and scraping tools
    "web_search",
    "search_with_filters",
    "search_news",
    "search_academic",
    "scrape_webpage",
    "extract_article_content",
    "scrape_with_links",
    # File operations
    "read_file_content",
    "write_file_content",
    "list_directory_contents",
    # Data processing
    "analyze_data",
    "format_output",
    "extract_key_information",
    # Domain-specific tools
    "get_weather",
    "send_slack_message",
    "summarize_text",
    # Automated search pipelines (no LLM processing)
    "search_and_scrape_pipeline",
    "wikipedia_search_pipeline",
    "multi_source_search_pipeline",
]
