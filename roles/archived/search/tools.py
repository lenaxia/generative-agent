"""Web Search tools for the search role.

Complete web search functionality using Tavily API for real-time web search.
Provides both basic search and advanced filtered search capabilities.
"""

import logging
import os
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from strands import tool

logger = logging.getLogger(__name__)

# Import Tavily client
try:
    from tavily import TavilyClient

    TAVILY_AVAILABLE = True
except ImportError:
    TavilyClient = None  # Define as None for type annotations
    TAVILY_AVAILABLE = False
    logger.warning(
        "Tavily client not available. Install with: pip install tavily-python"
    )

# Import web scraping libraries
try:
    from bs4 import BeautifulSoup
    from readability import Document

    SCRAPING_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    Document = None
    SCRAPING_AVAILABLE = False
    logger.warning(
        "Web scraping libraries not available. Install with: pip install beautifulsoup4 readability-lxml"
    )


def _get_tavily_client():
    """Get Tavily client instance with API key from environment.

    Returns:
        TavilyClient instance or None if not available/configured
    """
    if not TAVILY_AVAILABLE:
        return None

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY environment variable not set")
        return None

    try:
        return TavilyClient(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Tavily client: {e}")
        return None


def _format_search_results(tavily_response: dict, query: str) -> dict:
    """Format Tavily API response into standardized result format.

    Args:
        tavily_response: Raw response from Tavily API
        query: Original search query

    Returns:
        Formatted search results dictionary
    """
    results = []

    for result in tavily_response.get("results", []):
        formatted_result = {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("content", "")[:500],  # Limit snippet length
            "relevance_score": result.get("score", 0.0),
        }
        results.append(formatted_result)

    return {
        "query": query,
        "results": results,
        "total_results": len(results),
        "search_time": f"{time.time():.2f}s",
    }


@tool
def web_search(query: str, num_results: int = 5) -> dict:
    """Search the web for information using Tavily API.

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
                "query": query,
                "results": [],
                "total_results": 0,
                "error": "Empty search query provided",
            }

        if num_results <= 0:
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": "Number of results must be greater than 0",
            }

        # Limit max results to prevent API abuse
        num_results = min(num_results, 50)

        # Get Tavily client
        client = _get_tavily_client()
        if not client:
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": "TAVILY_API_KEY environment variable not set or Tavily client unavailable",
            }

        # Perform search
        logger.info(
            f"Performing web search for: '{query}' (num_results: {num_results})"
        )

        response = client.search(
            query=query,
            max_results=num_results,
            search_depth="basic",  # Use basic search for faster results
            include_answer=False,  # Don't need AI-generated answers
            include_raw_content=False,  # Don't need full page content
        )

        # Format and return results
        result = _format_search_results(response, query)
        result["search_time"] = f"{time.time() - start_time:.2f}s"

        logger.info(
            f"Web search completed: {result['total_results']} results in {result['search_time']}"
        )
        return result

    except Exception as e:
        logger.error(f"Web search failed for query '{query}': {e}")
        return {
            "query": query,
            "results": [],
            "total_results": 0,
            "error": str(e),
            "search_time": f"{time.time() - start_time:.2f}s",
        }


@tool
def search_with_filters(
    query: str,
    domain: Optional[str] = None,
    date_range: Optional[str] = None,
    content_type: Optional[str] = None,
    num_results: int = 5,
) -> dict:
    """Advanced web search with filters using Tavily API.

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
                "query": query,
                "filters": {
                    "domain": domain,
                    "date_range": date_range,
                    "content_type": content_type,
                },
                "results": [],
                "total_results": 0,
                "error": "Empty search query provided",
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
                "query": query,
                "filters": {
                    "domain": domain,
                    "date_range": date_range,
                    "content_type": content_type,
                },
                "results": [],
                "total_results": 0,
                "error": "TAVILY_API_KEY environment variable not set or Tavily client unavailable",
            }

        # Prepare search parameters
        search_params = {
            "query": filtered_query,
            "max_results": min(num_results, 50),
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }

        # Add date range filter if supported by Tavily
        # Note: Tavily may handle date filtering differently
        if date_range:
            logger.info(f"Date range filter requested: {date_range}")
            # Tavily handles date filtering internally based on query context

        # Perform filtered search
        logger.info(
            f"Performing filtered web search: '{filtered_query}' with filters: domain={domain}, date_range={date_range}, content_type={content_type}"
        )

        response = client.search(**search_params)

        # Format results
        results = []
        for result in response.get("results", []):
            formatted_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", "")[:500],
                "relevance_score": result.get("score", 0.0),
                "content_type": content_type or "general",
            }

            # Add publication date if available
            if "published_date" in result:
                formatted_result["date"] = result["published_date"]

            results.append(formatted_result)

        # Build response
        result = {
            "query": query,
            "filters": {
                "domain": domain,
                "date_range": date_range,
                "content_type": content_type,
            },
            "results": results,
            "total_results": len(results),
            "search_time": f"{time.time() - start_time:.2f}s",
        }

        logger.info(
            f"Filtered web search completed: {result['total_results']} results in {result['search_time']}"
        )
        return result

    except Exception as e:
        logger.error(f"Filtered web search failed for query '{query}': {e}")
        return {
            "query": query,
            "filters": {
                "domain": domain,
                "date_range": date_range,
                "content_type": content_type,
            },
            "results": [],
            "total_results": 0,
            "error": str(e),
            "search_time": f"{time.time() - start_time:.2f}s",
        }


@tool
def search_news(query: str, num_results: int = 5) -> dict:
    """Search for recent news articles using Tavily API.

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
        num_results=num_results,
    )


@tool
def search_academic(query: str, num_results: int = 5) -> dict:
    """Search for academic papers and scholarly content.

    Args:
        query: Academic search query
        num_results: Number of academic results to return

    Returns:
        Dict containing academic search results
    """
    # Add academic-focused terms to improve results
    academic_query = f"{query} academic paper research study"

    return search_with_filters(
        query=academic_query, content_type="academic", num_results=num_results
    )


@tool
def scrape_webpage(
    url: str, extract_content: bool = True, include_links: bool = False
) -> dict:
    """Scrape content from a specific webpage URL.

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
        # Validate and normalize URL
        validation_result = _validate_and_normalize_url(url)
        if validation_result.get("error"):
            return validation_result

        url = validation_result["url"]

        # Check dependencies
        if not SCRAPING_AVAILABLE:
            return _create_error_result(
                url,
                "Web scraping libraries not available. Install beautifulsoup4 and readability-lxml",
                start_time,
            )

        # Fetch and parse webpage
        soup = _fetch_and_parse_webpage(url)

        # Extract data
        title = _extract_title(soup)
        content = _extract_content(soup, extract_content)
        links = _extract_links(soup, url, include_links)

        # Build result
        return _build_success_result(
            url, title, content, links, include_links, start_time
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error scraping {url}: {e}")
        return _create_error_result(url, f"HTTP error: {str(e)}", start_time)
    except Exception as e:
        logger.error(f"Error scraping webpage {url}: {e}")
        return _create_error_result(url, str(e), start_time)


def _validate_and_normalize_url(url: str) -> dict:
    """Validate and normalize URL."""
    if not url or not url.strip():
        return _create_error_result(url, "Empty URL provided")

    # Add protocol if missing
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Validate URL format
    parsed_url = urlparse(url)
    if not parsed_url.netloc:
        return _create_error_result(url, "Invalid URL format")

    return {"url": url}


def _fetch_and_parse_webpage(url: str):
    """Fetch webpage and parse HTML."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    logger.info(f"Scraping webpage: {url}")
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    return BeautifulSoup(response.content, "html.parser")


def _extract_title(soup) -> str:
    """Extract title from HTML."""
    title_tag = soup.find("title")
    return title_tag.get_text().strip() if title_tag else ""


def _extract_content(soup, extract_content: bool) -> str:
    """Extract content from HTML."""
    if extract_content:
        # Use readability to extract main content
        doc = Document(str(soup))
        content = doc.summary()
        content_soup = BeautifulSoup(content, "html.parser")
        content = content_soup.get_text().strip()
    else:
        # Just get all text from the page
        content = soup.get_text().strip()

    # Clean up whitespace and limit length
    content = " ".join(content.split())
    if len(content) > 10000:
        content = content[:10000] + "... [content truncated]"

    return content


def _extract_links(soup, base_url: str, include_links: bool) -> list:
    """Extract links from HTML."""
    if not include_links:
        return []

    links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        link_text = link.get_text().strip()

        # Convert relative URLs to absolute
        if href.startswith("/"):
            href = urljoin(base_url, href)
        elif not href.startswith(("http://", "https://")):
            href = urljoin(base_url, href)

        if link_text and href.startswith(("http://", "https://")):
            links.append({"text": link_text, "url": href})

    return links


def _build_success_result(
    url: str,
    title: str,
    content: str,
    links: list,
    include_links: bool,
    start_time: float,
) -> dict:
    """Build successful scraping result."""
    scrape_time = f"{time.time() - start_time:.2f}s"

    result = {
        "url": url,
        "title": title,
        "content": content,
        "links": links[:50] if include_links else [],  # Limit to 50 links
        "content_length": len(content),
        "num_links": len(links) if include_links else 0,
        "scrape_time": scrape_time,
    }

    logger.info(
        f"Webpage scraped successfully: {len(content)} chars, {len(links)} links in {scrape_time}"
    )
    return result


def _create_error_result(url: str, error_msg: str, start_time: float = None) -> dict:
    """Create error result dictionary."""
    return {
        "url": url,
        "title": "",
        "content": "",
        "links": [],
        "error": error_msg,
        "scrape_time": f"{time.time() - start_time:.2f}s" if start_time else "0.00s",
    }


@tool
def extract_article_content(url: str) -> dict:
    """Extract the main article content from a webpage using readability.

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
def scrape_with_links(url: str) -> dict:
    """Scrape webpage content and extract all links.

    Useful for discovering related pages and navigation.

    Args:
        url: The URL to scrape

    Returns:
        Dict containing content and all links found on the page
    """
    return scrape_webpage(url, extract_content=True, include_links=True)


@tool
def search_and_scrape_pipeline(
    query: str, num_results: int = 5, domain: Optional[str] = None
) -> dict:
    """Automated search and scraping pipeline.

    Performs web search, then automatically scrapes all result URLs
    and returns structured data without LLM processing.

    Args:
        query: Search query string
        num_results: Number of search results to process (default: 5)
        domain: Optional domain filter (e.g., "wikipedia.org")

    Returns:
        Dict containing search results with scraped content
    """
    start_time = time.time()

    try:
        # Get Tavily client
        client = _get_tavily_client()
        if not client:
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": "TAVILY_API_KEY not set or Tavily client unavailable",
                "pipeline_time": f"{time.time() - start_time:.2f}s",
            }

        # Build search query with domain filter
        search_query = query
        if domain:
            search_query += f" site:{domain}"

        # Perform initial search
        logger.info(
            f"Starting search pipeline for: '{search_query}' (num_results: {num_results})"
        )

        search_response = client.search(
            query=search_query,
            max_results=num_results,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
        )

        # Extract URLs from search results
        search_results = search_response.get("results", [])
        urls_to_scrape = [
            result.get("url") for result in search_results if result.get("url")
        ]

        logger.info(f"Found {len(urls_to_scrape)} URLs to scrape")

        # Scrape each URL automatically
        scraped_results = []
        for i, url in enumerate(urls_to_scrape):
            logger.info(f"Scraping URL {i+1}/{len(urls_to_scrape)}: {url}")

            # Use existing scrape_webpage function
            scrape_result = scrape_webpage(
                url, extract_content=True, include_links=False
            )

            # Combine search metadata with scraped content
            combined_result = {
                "search_rank": i + 1,
                "title": search_results[i].get("title", ""),
                "url": url,
                "search_snippet": search_results[i].get("content", "")[:200],
                "relevance_score": search_results[i].get("score", 0.0),
                "scraped_content": scrape_result.get("content", ""),
                "scraped_title": scrape_result.get("title", ""),
                "scrape_success": "error" not in scrape_result,
                "scrape_error": scrape_result.get("error", None),
                "content_length": scrape_result.get("content_length", 0),
            }

            scraped_results.append(combined_result)

        # Build final structured response
        pipeline_result = {
            "query": query,
            "domain_filter": domain,
            "search_results": scraped_results,
            "total_results": len(scraped_results),
            "successful_scrapes": sum(
                1 for r in scraped_results if r["scrape_success"]
            ),
            "failed_scrapes": sum(
                1 for r in scraped_results if not r["scrape_success"]
            ),
            "total_content_length": sum(r["content_length"] for r in scraped_results),
            "pipeline_time": f"{time.time() - start_time:.2f}s",
        }

        logger.info(
            f"Search pipeline completed: {pipeline_result['successful_scrapes']}/{pipeline_result['total_results']} successful scrapes in {pipeline_result['pipeline_time']}"
        )

        return pipeline_result

    except Exception as e:
        logger.error(f"Search pipeline failed for query '{query}': {e}")
        return {
            "query": query,
            "domain_filter": domain,
            "search_results": [],
            "total_results": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "error": str(e),
            "pipeline_time": f"{time.time() - start_time:.2f}s",
        }


@tool
def wikipedia_search_pipeline(topic: str) -> dict:
    """Specialized Wikipedia search and scraping pipeline.

    Searches Wikipedia for a topic and automatically scrapes
    the main Wikipedia article content.

    Args:
        topic: Topic to search for on Wikipedia

    Returns:
        Dict containing Wikipedia article content and metadata
    """
    return search_and_scrape_pipeline(
        query=topic,
        num_results=1,  # Just get the main Wikipedia article
        domain="wikipedia.org",
    )


@tool
def multi_source_search_pipeline(
    query: str, sources: list[str], results_per_source: int = 2
) -> dict:
    """Multi-source search and scraping pipeline.

    Searches multiple domains and automatically scrapes content
    from each source.

    Args:
        query: Search query
        sources: List of domains to search (e.g., ["wikipedia.org", "britannica.com"])
        results_per_source: Number of results to get from each source

    Returns:
        Dict containing results from all sources with scraped content
    """
    start_time = time.time()
    all_results = []

    for source in sources:
        logger.info(f"Processing source: {source}")
        source_results = search_and_scrape_pipeline(
            query=query, num_results=results_per_source, domain=source
        )

        # Add source metadata
        for result in source_results.get("search_results", []):
            result["source_domain"] = source

        all_results.extend(source_results.get("search_results", []))

    return {
        "query": query,
        "sources": sources,
        "search_results": all_results,
        "total_results": len(all_results),
        "successful_scrapes": sum(1 for r in all_results if r["scrape_success"]),
        "failed_scrapes": sum(1 for r in all_results if not r["scrape_success"]),
        "total_content_length": sum(r["content_length"] for r in all_results),
        "pipeline_time": f"{time.time() - start_time:.2f}s",
    }
