"""
Search Pipeline Tool

Automated search and scraping pipeline that performs web search,
automatically scrapes all result URLs, and returns structured data
without LLM processing.
"""

from strands import tool
from typing import Dict, List, Optional
import logging
import time
from roles.shared_tools.web_search import _get_tavily_client, scrape_webpage

logger = logging.getLogger(__name__)


@tool
def search_and_scrape_pipeline(query: str, num_results: int = 5, 
                              domain: Optional[str] = None) -> Dict:
    """
    Automated search and scraping pipeline.
    
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
                'query': query,
                'results': [],
                'total_results': 0,
                'error': 'TAVILY_API_KEY not set or Tavily client unavailable',
                'pipeline_time': f"{time.time() - start_time:.2f}s"
            }
        
        # Build search query with domain filter
        search_query = query
        if domain:
            search_query += f" site:{domain}"
        
        # Perform initial search
        logger.info(f"Starting search pipeline for: '{search_query}' (num_results: {num_results})")
        
        search_response = client.search(
            query=search_query,
            max_results=num_results,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False
        )
        
        # Extract URLs from search results
        search_results = search_response.get('results', [])
        urls_to_scrape = [result.get('url') for result in search_results if result.get('url')]
        
        logger.info(f"Found {len(urls_to_scrape)} URLs to scrape")
        
        # Scrape each URL automatically
        scraped_results = []
        for i, url in enumerate(urls_to_scrape):
            logger.info(f"Scraping URL {i+1}/{len(urls_to_scrape)}: {url}")
            
            # Use existing scrape_webpage function
            scrape_result = scrape_webpage(url, extract_content=True, include_links=False)
            
            # Combine search metadata with scraped content
            combined_result = {
                'search_rank': i + 1,
                'title': search_results[i].get('title', ''),
                'url': url,
                'search_snippet': search_results[i].get('content', '')[:200],
                'relevance_score': search_results[i].get('score', 0.0),
                'scraped_content': scrape_result.get('content', ''),
                'scraped_title': scrape_result.get('title', ''),
                'scrape_success': 'error' not in scrape_result,
                'scrape_error': scrape_result.get('error', None),
                'content_length': scrape_result.get('content_length', 0)
            }
            
            scraped_results.append(combined_result)
        
        # Build final structured response
        pipeline_result = {
            'query': query,
            'domain_filter': domain,
            'search_results': scraped_results,
            'total_results': len(scraped_results),
            'successful_scrapes': sum(1 for r in scraped_results if r['scrape_success']),
            'failed_scrapes': sum(1 for r in scraped_results if not r['scrape_success']),
            'total_content_length': sum(r['content_length'] for r in scraped_results),
            'pipeline_time': f"{time.time() - start_time:.2f}s"
        }
        
        logger.info(f"Search pipeline completed: {pipeline_result['successful_scrapes']}/{pipeline_result['total_results']} successful scrapes in {pipeline_result['pipeline_time']}")
        
        return pipeline_result
        
    except Exception as e:
        logger.error(f"Search pipeline failed for query '{query}': {e}")
        return {
            'query': query,
            'domain_filter': domain,
            'search_results': [],
            'total_results': 0,
            'successful_scrapes': 0,
            'failed_scrapes': 0,
            'error': str(e),
            'pipeline_time': f"{time.time() - start_time:.2f}s"
        }


@tool
def wikipedia_search_pipeline(topic: str) -> Dict:
    """
    Specialized Wikipedia search and scraping pipeline.
    
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
        domain="wikipedia.org"
    )


@tool
def multi_source_search_pipeline(query: str, sources: List[str], 
                                 results_per_source: int = 2) -> Dict:
    """
    Multi-source search and scraping pipeline.
    
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
            query=query,
            num_results=results_per_source,
            domain=source
        )
        
        # Add source metadata
        for result in source_results.get('search_results', []):
            result['source_domain'] = source
        
        all_results.extend(source_results.get('search_results', []))
    
    return {
        'query': query,
        'sources': sources,
        'search_results': all_results,
        'total_results': len(all_results),
        'successful_scrapes': sum(1 for r in all_results if r['scrape_success']),
        'failed_scrapes': sum(1 for r in all_results if not r['scrape_success']),
        'total_content_length': sum(r['content_length'] for r in all_results),
        'pipeline_time': f"{time.time() - start_time:.2f}s"
    }