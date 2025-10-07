"""
Search Data Collector Role

Programmatic role for pure data collection without analysis.
Designed to eliminate redundant LLM analysis calls in search → analysis workflows.

This role:
1. Uses LLM for natural language instruction parsing
2. Executes search pipeline programmatically
3. Returns structured data WITHOUT analysis
4. Supports intelligent follow-up searches when needed
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from common.task_context import TaskContext
from llm_provider.factory import LLMType
from llm_provider.programmatic_role import ProgrammaticRole

logger = logging.getLogger(__name__)


class SearchPipelineTools:
    """Mock search pipeline tools for the SearchDataCollectorRole."""

    def search_source(
        self, query: str, source: str, num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Mock search function that would integrate with actual search tools.

        In real implementation, this would call:
        - web_search() for web source
        - wikipedia search for wikipedia source
        - academic search for academic source
        """
        # Mock implementation - in real version would call actual search tools
        return [
            {
                "title": f"{source.title()} result for {query}",
                "url": f"http://{source}.example.com/result",
                "snippet": f"Mock {source} content about {query}",
                "source": source,
            }
        ] * min(
            num_results, 2
        )  # Return up to 2 mock results per source


class SearchDataCollectorRole(ProgrammaticRole):
    """
    Programmatic role for pure data collection without analysis.

    Key features:
    - LLM-assisted instruction parsing
    - Pure programmatic data collection
    - No analysis or summarization
    - Intelligent follow-up searches
    - Minimal LLM usage (1-2 calls max)
    """

    def __init__(self):
        super().__init__(
            name="search_data_collector",
            description="Collects raw search data without analysis for downstream processing",
        )
        self.search_tools = SearchPipelineTools()
        self.llm_factory = None  # Will be injected

    def execute(
        self, instruction: str, context: Optional[TaskContext] = None
    ) -> Dict[str, Any]:
        """
        Execute search with LLM parsing but no analysis.

        Args:
            instruction: Natural language search instruction
            context: Optional task context

        Returns:
            Dict containing structured search results without analysis
        """
        start_time = time.time()

        try:
            # Step 1: LLM parses natural language to structured parameters
            params = self._llm_parse_search_instruction(instruction)

            # Step 2: Pure programmatic data collection (no LLM analysis)
            raw_results = self._execute_search_pipeline(params)

            # Step 3: Optional LLM-guided follow-ups if needed
            llm_calls = 1  # Initial parsing call
            if self._needs_followup(raw_results, params):
                followup_params = self._llm_determine_followup(raw_results, params)
                if followup_params:
                    additional_results = self._execute_search_pipeline(followup_params)
                    raw_results.extend(additional_results)
                    llm_calls += 1  # Follow-up call

            # Update metrics
            execution_time = time.time() - start_time
            self._track_execution_time(start_time)

            # Return structured data WITHOUT analysis
            return {
                "search_results": raw_results,
                "metadata": {
                    "query": params["query"],
                    "sources_searched": params["sources"],
                    "total_results": len(raw_results),
                    "search_timestamp": time.time(),
                    "llm_calls": llm_calls,
                },
                "execution_type": "programmatic_data_collection",
            }

        except Exception as e:
            logger.error(f"Search data collection failed: {e}")
            execution_time = time.time() - start_time
            return self._create_error_result(e, execution_time)

    def parse_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Parse instruction to extract search parameters.

        Args:
            instruction: Raw instruction string

        Returns:
            Dict: Parsed parameters for search execution
        """
        return self._llm_parse_search_instruction(instruction)

    def _llm_parse_search_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Use LLM to convert natural language to structured parameters.

        Args:
            instruction: Natural language instruction

        Returns:
            Dict: Structured search parameters
        """
        parsing_prompt = f"""
        Convert this search instruction to structured parameters:
        Instruction: "{instruction}"

        Return JSON with these fields:
        {{
            "query": "main search terms",
            "sources": ["web", "wikipedia", "academic"],
            "num_results": 5,
            "focus": "specific aspect to focus on",
            "min_results": 3
        }}

        Be precise and extract all relevant parameters.
        """

        # Use WEAK model for cost-effective parsing
        if self.llm_factory:
            agent = self.llm_factory.create_agent(LLMType.WEAK)
            response = agent(parsing_prompt)
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response as JSON: {response}")
                # Fallback to simple parsing
                return self._fallback_parse_instruction(instruction)
        else:
            # Fallback when no LLM factory available
            return self._fallback_parse_instruction(instruction)

    def _fallback_parse_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Fallback instruction parsing without LLM.

        Args:
            instruction: Natural language instruction

        Returns:
            Dict: Basic parsed parameters
        """
        return {
            "query": instruction,
            "sources": ["web", "wikipedia"],
            "num_results": 5,
            "focus": "general information",
            "min_results": 3,
        }

    def _execute_search_pipeline(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute search without any LLM analysis - pure data collection.

        Args:
            params: Structured search parameters

        Returns:
            List of raw search results
        """
        results = []
        for source in params["sources"]:
            try:
                source_results = self.search_tools.search_source(
                    query=params["query"],
                    source=source,
                    num_results=params.get("num_results", 5),
                )
                results.extend(source_results)
            except Exception as e:
                logger.warning(f"Failed to search source '{source}': {e}")
                # Continue with other sources

        # Return raw structured data - no summarization, no analysis
        return results

    def _needs_followup(self, results: List[Dict], params: Dict) -> bool:
        """
        Determine if follow-up searches are needed.

        Args:
            results: Current search results
            params: Original search parameters

        Returns:
            bool: True if follow-up search is needed
        """
        # Rule-based checks
        min_results = params.get("min_results", 3)
        if len(results) < min_results:
            return True

        # Could add more sophisticated checks here
        return False

    def _llm_determine_followup(
        self, results: List[Dict], params: Dict
    ) -> Optional[Dict]:
        """
        Use LLM to determine follow-up search parameters if needed.

        Args:
            results: Current search results
            params: Original search parameters

        Returns:
            Dict: Follow-up search parameters or None
        """
        if not self._needs_followup(results, params):
            return None

        followup_prompt = f"""
        Based on these search results, determine if follow-up search is needed:

        Original query: {params["query"]}
        Results found: {len(results)}
        Minimum needed: {params.get("min_results", 3)}

        Should we search with different terms? Return JSON with new parameters or null.
        Format: {{"query": "new search terms", "sources": ["source1"], "num_results": 3}}
        """

        if self.llm_factory:
            try:
                agent = self.llm_factory.create_agent(LLMType.WEAK)
                response = agent(followup_prompt)

                if response.strip().lower() == "null":
                    return None

                return json.loads(response)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to determine follow-up parameters: {e}")
                return None

        return None
