"""Data Collection Role

Programmatic role for structured data collection from multiple sources.
Designed for pure automation workflows without any LLM processing.

This role:
1. Performs rule-based instruction parsing
2. Executes multi-source data collection programmatically
3. Returns structured data without any analysis
4. Supports parallel data collection for efficiency
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from common.task_context import TaskContext
from llm_provider.programmatic_role import ProgrammaticRole

logger = logging.getLogger(__name__)


class DataSourceManager:
    """Mock data source manager for the DataCollectionRole."""

    def collect_from_source(
        self, source: str, data_types: list[str]
    ) -> list[dict[str, Any]]:
        """Mock data collection from a specific source.

        In real implementation, this would integrate with:
        - Database connections
        - REST API clients
        - File system operations
        - Cloud storage services
        """
        # Mock implementation - in real version would call actual data sources
        mock_data = []

        for data_type in data_types:
            mock_data.extend(
                [
                    {
                        "id": f"{source}_{data_type}_{i}",
                        "type": data_type,
                        "data": f"Mock {data_type} data from {source}",
                        "source": source,
                        "timestamp": time.time(),
                    }
                    for i in range(2)  # 2 items per data type
                ]
            )

        return mock_data


class DataCollectionRole(ProgrammaticRole):
    """Programmatic role for structured data collection from multiple sources.

    Key features:
    - Rule-based instruction parsing (no LLM needed)
    - Multi-source data aggregation
    - Parallel data collection for efficiency
    - Structured data formatting and validation
    - Zero LLM calls - pure automation
    """

    def __init__(self):
        """Initialize the DataCollectionRole with data source management.

        Sets up the role with multi-source data aggregation capabilities
        and initializes the data source manager for automation workflows.
        """
        super().__init__(
            name="data_collection",
            description="Multi-source data aggregation and structured processing for automation workflows",
        )
        self.data_sources = DataSourceManager()

    def execute(
        self, instruction: str, context: Optional[TaskContext] = None
    ) -> dict[str, Any]:
        """Execute multi-source data collection.

        Args:
            instruction: Data collection instruction
            context: Optional task context

        Returns:
            Dict containing collected data from all sources
        """
        start_time = time.time()

        try:
            # Parse instruction using rule-based parsing (no LLM)
            params = self.parse_instruction(instruction)

            # Execute parallel data collection
            collected_data = self._collect_data_parallel(params)

            # Calculate total records
            total_records = sum(len(data) for data in collected_data.values())

            # Update metrics
            self._track_execution_time(start_time)

            # Return structured data without any analysis
            return {
                "collected_data": collected_data,
                "sources": params["sources"],
                "total_records": total_records,
                "execution_metadata": {
                    "role": self.name,
                    "execution_type": "programmatic",
                    "execution_time": f"{time.time() - start_time:.2f}s",
                    "llm_calls": 0,  # Zero LLM calls - pure automation
                    "sources_processed": len(params["sources"]),
                    "data_types_collected": len(params["data_types"]),
                },
            }

        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            execution_time = time.time() - start_time
            return self._create_error_result(e, execution_time)

    def parse_instruction(self, instruction: str) -> dict[str, Any]:
        """Parse data collection parameters using rule-based parsing.

        Args:
            instruction: Raw instruction string

        Returns:
            Dict: Parsed parameters for data collection
        """
        # Rule-based parsing - no LLM needed
        sources = self._extract_sources(instruction)
        data_types = self._extract_data_types(instruction)
        format_type = self._extract_format(instruction)

        return {
            "sources": sources,
            "data_types": data_types,
            "format": format_type,
            "parallel": True,  # Enable parallel collection by default
        }

    def _extract_sources(self, instruction: str) -> list[str]:
        """Extract data sources from instruction using pattern matching."""
        instruction_lower = instruction.lower()
        sources = []

        # Common source patterns
        source_patterns = {
            "database": ["database", "db", "sql", "mysql", "postgres"],
            "api": ["api", "rest", "endpoint", "service"],
            "file": ["file", "csv", "json", "xml", "document"],
            "cloud": ["cloud", "s3", "azure", "gcs", "storage"],
            "cache": ["cache", "redis", "memcache"],
            "queue": ["queue", "kafka", "rabbitmq", "sqs"],
        }

        for source_type, patterns in source_patterns.items():
            if any(pattern in instruction_lower for pattern in patterns):
                sources.append(source_type)

        # Default sources if none detected
        if not sources:
            sources = ["database", "api"]

        return sources

    def _extract_data_types(self, instruction: str) -> list[str]:
        """Extract data types from instruction using pattern matching."""
        instruction_lower = instruction.lower()
        data_types = []

        # Common data type patterns
        type_patterns = {
            "user": ["user", "customer", "account", "profile"],
            "product": ["product", "item", "catalog", "inventory"],
            "order": ["order", "transaction", "purchase", "sale"],
            "log": ["log", "event", "activity", "audit"],
            "metric": ["metric", "analytics", "stats", "measurement"],
            "config": ["config", "setting", "parameter", "option"],
        }

        for data_type, patterns in type_patterns.items():
            if any(pattern in instruction_lower for pattern in patterns):
                data_types.append(data_type)

        # Default data types if none detected
        if not data_types:
            data_types = ["user", "product"]

        return data_types

    def _extract_format(self, instruction: str) -> str:
        """Extract output format from instruction."""
        instruction_lower = instruction.lower()

        if "json" in instruction_lower:
            return "json"
        elif "csv" in instruction_lower:
            return "csv"
        elif "xml" in instruction_lower:
            return "xml"
        else:
            return "json"  # Default format

    def _collect_data_parallel(
        self, params: dict[str, Any]
    ) -> dict[str, list[dict[str, Any]]]:
        """Collect data from multiple sources in parallel for efficiency.

        Args:
            params: Collection parameters

        Returns:
            Dict mapping source names to collected data
        """
        collected_data = {}

        if params.get("parallel", True) and len(params["sources"]) > 1:
            # Parallel collection for multiple sources
            with ThreadPoolExecutor(
                max_workers=min(len(params["sources"]), 5)
            ) as executor:
                future_to_source = {
                    executor.submit(
                        self._collect_from_single_source, source, params["data_types"]
                    ): source
                    for source in params["sources"]
                }

                for future in as_completed(future_to_source):
                    source = future_to_source[future]
                    try:
                        data = future.result()
                        collected_data[source] = data
                    except Exception as e:
                        logger.warning(f"Failed to collect from source '{source}': {e}")
                        collected_data[source] = []  # Empty data for failed source
        else:
            # Sequential collection
            for source in params["sources"]:
                try:
                    data = self._collect_from_single_source(
                        source, params["data_types"]
                    )
                    collected_data[source] = data
                except Exception as e:
                    logger.warning(f"Failed to collect from source '{source}': {e}")
                    collected_data[source] = []

        return collected_data

    def _collect_from_single_source(
        self, source: str, data_types: list[str]
    ) -> list[dict[str, Any]]:
        """Collect data from a single source.

        Args:
            source: Source name
            data_types: List of data types to collect

        Returns:
            List of collected data items
        """
        return self.data_sources.collect_from_source(source, data_types)
