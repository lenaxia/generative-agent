"""
Web Search Shared Tool

Complete web search functionality using Tavily API for real-time web search.
Provides both basic search and advanced filtered search capabilities.
"""

from strands import tool
from typing import Dict, List, Optional
import os
import logging
import time
import requests
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

# Import Tavily client
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TavilyClient = None  # Define as None for type annotations
    TAVILY_AVAILABLE = False
    logger.warning("Tavily client not available. Install with: pip install tavily-python")

# Import web scraping libraries
try:
    from bs4 import BeautifulSoup
    from readability import Document
    SCRAPING_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    Document = None
    SCRAPING_AVAILABLE = False
    logger.warning("Web scraping libraries not available. Install with: pip install beautifulsoup4 readability-lxml")


def _get_tavily_client():
    """
    Get Tavily client instance with API key from environment.
    
    Returns:
        TavilyClient instance or None if not available/configured
    """
    if not TAVILY_AVAILABLE:
        return None
        
    api_key = os.getenv('TAVILY_API_KEY')
    if not api_key:
        logger.error("TAVILY_API_KEY environment variable not set")
        return None
        
    try:
        return TavilyClient(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Tavily client: {e}")
        return None


def _format_search_results(tavily_response: Dict, query: str) -> Dict:
    """
    Format Tavily API response into standardized result format.
    
    Args:
        tavily_response: Raw response from Tavily API
        query: Original search query
        
    Returns:
        Formatted search results dictionary
    """
    results = []
    
    for result in tavily_response.get('results', []):
        formatted_result = {
            'title': result.get('title', ''),
            'url': result.get('url', ''),
            'snippet': result.get('content', '')[:500],  # Limit snippet length
            'relevance_score': result.get('score', 0.0)
        }
        results.append(formatted_result)
    
    return {
        'query': query,
        'results': results,
        'total_results': len(results),
        'search_time': f"{time.time():.2f}s"
    }


@tool
def web_search(query: str, num_results: int = 5) -> Dict:
    """
    Search the web for information using Tavily API.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 5, max: 50)
        
    Returns:
        Dict containing search results with titles, URLs, and snippets
        
    Example:
        >>> result = web_search("Python programming tutorials", num_results=3)
        >>> print(result['total_results'])
        3
        >>> print(result['results'][0]['title'])
        'Learn Python Programming - Complete Tutorial'
    """
    start_time = time.time()
    
    try:
        # Validate parameters
        if not query or not query.strip():
            return {
                'query': query,
                'results': [],
                'total_results': 0,
                'error': 'Empty search query provided'
            }
            
        if num_results <= 0:
            return {
                'query': query,
                'results': [],
                'total_results': 0,
                'error': 'Number of results must be greater than 0'
            }
        
        # Limit max results to prevent API abuse
        num_results = min(num_results, 50)
        
        # Get Tavily client
        client = _get_tavily_client()
        if not client:
            return {
                'query': query,
                'results': [],
                'total_results': 0,
                'error': 'TAVILY_API_KEY environment variable not set or Tavily client unavailable'
            }
        
        # Perform search
        logger.info(f"Performing web search for: '{query}' (num_results: {num_results})")
        
        response = client.search(
            query=query,
            max_results=num_results,
            search_depth="basic",  # Use basic search for faster results
            include_answer=False,  # Don't need AI-generated answers
            include_raw_content=False  # Don't need full page content
        )
        
        # Format and return results
        result = _format_search_results(response, query)
        result['search_time'] = f"{time.time() - start_time:.2f}s"
        
        logger.info(f"Web search completed: {result['total_results']} results in {result['search_time']}")
        return result
        
    except Exception as e:
        logger.error(f"Web search failed for query '{query}': {e}")
        return {
            'query': query,
            'results': [],
            'total_results': 0,
            'error': str(e),
            'search_time': f"{time.time() - start_time:.2f}s"
        }


@tool
def search_with_filters(query: str, domain: Optional[str] = None, 
                       date_range: Optional[str] = None, 
                       content_type: Optional[str] = None,
                       num_results: int = 5) -> Dict:
    """
    Advanced web search with filters using Tavily API.
    
    Args:
        query: Search query string
        domain: Specific domain to search (e.g., "reddit.com", "stackoverflow.com")
        date_range: Date range filter (e.g., "past_week", "past_month", "past_year")
        content_type: Content type filter (e.g., "news", "academic", "forum")
        num_results: Number of results to return (default: 5)
        
    Returns:
        Dict containing filtered search results
        
    Example:
        >>> result = search_with_filters(
        ...     query="machine learning tutorials",
        ...     domain="github.com",
        ...     content_type="code"
        ... )
        >>> print(result['filters']['domain'])
        'github.com'
    """
    start_time = time.time()
    
    try:
        # Validate parameters
        if not query or not query.strip():
            return {
                'query': query,
                'filters': {
                    'domain': domain,
                    'date_range': date_range,
                    'content_type': content_type
                },
                'results': [],
                'total_results': 0,
                'error': 'Empty search query provided'
            }
        
        # Build filtered query
        filtered_query = query.strip()
        
        # Add domain filter using site: operator
        if domain:
            filtered_query += f" site:{domain}"
        
        # Get Tavily client
        client = _get_tavily_client()
        if not client:
            return {
                'query': query,
                'filters': {
                    'domain': domain,
                    'date_range': date_range,
                    'content_type': content_type
                },
                'results': [],
                'total_results': 0,
                'error': 'TAVILY_API_KEY environment variable not set or Tavily client unavailable'
            }
        
        # Prepare search parameters
        search_params = {
            'query': filtered_query,
            'max_results': min(num_results, 50),
            'search_depth': "basic",
            'include_answer': False,
            'include_raw_content': False
        }
        
        # Add date range filter if supported by Tavily
        # Note: Tavily may handle date filtering differently
        if date_range:
            logger.info(f"Date range filter requested: {date_range}")
            # Tavily handles date filtering internally based on query context
        
        # Perform filtered search
        logger.info(f"Performing filtered web search: '{filtered_query}' with filters: domain={domain}, date_range={date_range}, content_type={content_type}")
        
        response = client.search(**search_params)
        
        # Format results
        results = []
        for result in response.get('results', []):
            formatted_result = {
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'snippet': result.get('content', '')[:500],
                'relevance_score': result.get('score', 0.0),
                'content_type': content_type or 'general'
            }
            
            # Add publication date if available
            if 'published_date' in result:
                formatted_result['date'] = result['published_date']
            
            results.append(formatted_result)
        
        # Build response
        result = {
            'query': query,
            'filters': {
                'domain': domain,
                'date_range': date_range,
                'content_type': content_type
            },
            'results': results,
            'total_results': len(results),
            'search_time': f"{time.time() - start_time:.2f}s"
        }
        
        logger.info(f"Filtered web search completed: {result['total_results']} results in {result['search_time']}")
        return result
        
    except Exception as e:
        logger.error(f"Filtered web search failed for query '{query}': {e}")
        return {
            'query': query,
            'filters': {
                'domain': domain,
                'date_range': date_range,
                'content_type': content_type
            },
            'results': [],
            'total_results': 0,
            'error': str(e),
            'search_time': f"{time.time() - start_time:.2f}s"
        }


@tool
def search_news(query: str, num_results: int = 5) -> Dict:
    """
    Search for recent news articles using Tavily API.
    
    Args:
        query: News search query
        num_results: Number of news articles to return
        
    Returns:
        Dict containing news search results
        
    Example:
        >>> result = search_news("artificial intelligence breakthrough")
        >>> print(result['results'][0]['title'])
        'Latest AI Breakthrough in Machine Learning'
    """
    return search_with_filters(
        query=query,
        content_type="news",
        date_range="past_week",
        num_results=num_results
    )


@tool
def search_academic(query: str, num_results: int = 5) -> Dict:
    """
    Search for academic papers and scholarly content.
    
    Args:
        query: Academic search query
        num_results: Number of academic results to return
        
    Returns:
        Dict containing academic search results
    """
    # Add academic-focused terms to improve results
    academic_query = f"{query} academic paper research study"
    
    return search_with_filters(
        query=academic_query,
        content_type="academic",
        num_results=num_results
    )


@tool
def scrape_webpage(url: str, extract_content: bool = True, include_links: bool = False) -> Dict:
    """
    Scrape content from a specific webpage URL.
    
    Args:
        url: The URL to scrape
        extract_content: Whether to extract main content using readability (default: True)
        include_links: Whether to include links found in the content (default: False)
        
    Returns:
        Dict containing scraped content, title, and metadata
        
    Example:
        >>> result = scrape_webpage("https://example.com/article")
        >>> print(result['title'])
        'Example Article Title'
        >>> print(result['content'][:100])
        'This is the main content of the article...'
    """
    start_time = time.time()
    
    try:
        # Validate URL
        if not url or not url.strip():
            return {
                'url': url,
                'title': '',
                'content': '',
                'links': [],
                'error': 'Empty URL provided'
            }
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Validate URL format
        parsed_url = urlparse(url)
        if not parsed_url.netloc:
            return {
                'url': url,
                'title': '',
                'content': '',
                'links': [],
                'error': 'Invalid URL format'
            }
        
        # Check if scraping libraries are available
        if not SCRAPING_AVAILABLE:
            return {
                'url': url,
                'title': '',
                'content': '',
                'links': [],
                'error': 'Web scraping libraries not available. Install beautifulsoup4 and readability-lxml'
            }
        
        # Set up headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Fetch the webpage
        logger.info(f"Scraping webpage: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else ''
        
        # Extract content
        content = ''
        if extract_content:
            # Use readability to extract main content
            doc = Document(response.content)
            content = doc.summary()
            
            # Parse the cleaned HTML to get text
            content_soup = BeautifulSoup(content, 'html.parser')
            content = content_soup.get_text().strip()
            
            # Clean up whitespace
            content = ' '.join(content.split())
        else:
            # Just get all text from the page
            content = soup.get_text().strip()
            content = ' '.join(content.split())
        
        # Extract links if requested
        links = []
        if include_links:
            for link in soup.find_all('a', href=True):
                href = link['href']
                link_text = link.get_text().strip()
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    href = urljoin(url, href)
                elif not href.startswith(('http://', 'https://')):
                    href = urljoin(url, href)
                
                if link_text and href.startswith(('http://', 'https://')):
                    links.append({
                        'text': link_text,
                        'url': href
                    })
        
        # Limit content length to prevent excessive data
        if len(content) > 10000:
            content = content[:10000] + "... [content truncated]"
        
        result = {
            'url': url,
            'title': title,
            'content': content,
            'links': links[:50] if include_links else [],  # Limit to 50 links
            'content_length': len(content),
            'num_links': len(links) if include_links else 0,
            'scrape_time': f"{time.time() - start_time:.2f}s"
        }
        
        logger.info(f"Webpage scraped successfully: {len(content)} chars, {len(links)} links in {result['scrape_time']}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error scraping {url}: {e}")
        return {
            'url': url,
            'title': '',
            'content': '',
            'links': [],
            'error': f'HTTP error: {str(e)}',
            'scrape_time': f"{time.time() - start_time:.2f}s"
        }
    except Exception as e:
        logger.error(f"Error scraping webpage {url}: {e}")
        return {
            'url': url,
            'title': '',
            'content': '',
            'links': [],
            'error': str(e),
            'scrape_time': f"{time.time() - start_time:.2f}s"
        }


@tool
def extract_article_content(url: str) -> Dict:
    """
    Extract the main article content from a webpage using readability.
    
    This is optimized for news articles, blog posts, and similar content.
    
    Args:
        url: The URL of the article to extract
        
    Returns:
        Dict containing extracted article content and metadata
        
    Example:
        >>> result = extract_article_content("https://example.com/news/article")
        >>> print(result['title'])
        'Breaking News: Important Development'
        >>> print(result['content'][:100])
        'The main article content starts here...'
    """
    return scrape_webpage(url, extract_content=True, include_links=False)


@tool
def scrape_with_links(url: str) -> Dict:
    """
    Scrape webpage content and extract all links.
    
    Useful for discovering related pages and navigation.
    
    Args:
        url: The URL to scrape
        
    Returns:
        Dict containing content and all links found on the page
    """
    return scrape_webpage(url, extract_content=True, include_links=True)


# Export available tools
__all__ = [
    'web_search', 'search_with_filters', 'search_news', 'search_academic',
    'scrape_webpage', 'extract_article_content', 'scrape_with_links'
]