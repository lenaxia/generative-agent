"""
Shared Tools Module

Common tools that can be used across multiple roles.
"""

from .web_search import web_search, search_with_filters, search_news, search_academic, scrape_webpage, extract_article_content, scrape_with_links
from .file_operations import read_file_content, write_file_content, list_directory_contents
from .data_processing import analyze_data, format_output, extract_key_information
from .weather_tools import get_weather
from .slack_tools import send_slack_message
from .summarizer_tools import summarize_text
from .search_pipeline import search_and_scrape_pipeline, wikipedia_search_pipeline, multi_source_search_pipeline

__all__ = [
    # Web search and scraping tools
    'web_search',
    'search_with_filters',
    'search_news',
    'search_academic',
    'scrape_webpage',
    'extract_article_content',
    'scrape_with_links',
    
    # File operations
    'read_file_content',
    'write_file_content',
    'list_directory_contents',
    
    # Data processing
    'analyze_data',
    'format_output',
    'extract_key_information',
    
    # Domain-specific tools
    'get_weather',
    'send_slack_message',
    'summarize_text',
    
    # Automated search pipelines (no LLM processing)
    'search_and_scrape_pipeline',
    'wikipedia_search_pipeline',
    'multi_source_search_pipeline',
]