"""
Programmatic Role Base Class

Abstract base class for programmatic roles that execute directly without LLM processing.
Designed for pure automation tasks, data collection, API integrations, and structured processing.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from common.task_context import TaskContext

logger = logging.getLogger(__name__)


class ProgrammaticRole(ABC):
    """
    Base class for programmatic roles that execute directly without LLM processing.
    
    Programmatic roles are designed for:
    - Pure automation tasks
    - Data collection and processing
    - API integrations
    - File operations
    - Structured data transformations
    
    Key characteristics:
    - Execute directly without LLM reasoning overhead
    - May use one LLM call for natural language parsing
    - Return structured data for downstream processing
    - Optimized for performance and cost efficiency
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize programmatic role with basic metadata.
        
        Args:
            name: Role name identifier
            description: Human-readable role description
        """
        self.name = name
        self.description = description
        self.execution_count = 0
        self.total_execution_time = 0.0
        
        logger.info(f"Initialized programmatic role: {name}")
    
    @abstractmethod
    def execute(self, instruction: str, context: Optional[TaskContext] = None) -> Any:
        """
        Execute the programmatic task directly.
        
        This method should implement the core logic of the role without
        relying on LLM reasoning. It may use one LLM call for instruction
        parsing if needed, but the main execution should be programmatic.
        
        Args:
            instruction: Task instruction (may contain parameters)
            context: Optional task context for state access
            
        Returns:
            Any: Structured result data (dict, list, str, etc.)
            
        Raises:
            Exception: Any execution errors should be propagated
        """
        pass
    
    @abstractmethod
    def parse_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Parse instruction to extract parameters for execution.
        
        This method should convert natural language instructions into
        structured parameters that can be used for programmatic execution.
        May use LLM for complex parsing or implement rule-based parsing.
        
        Args:
            instruction: Raw instruction string
            
        Returns:
            Dict: Parsed parameters for execution
            
        Raises:
            Exception: Parsing errors should be propagated
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get execution metrics for this role.
        
        Returns:
            Dict containing execution statistics and performance metrics
        """
        return {
            "name": self.name,
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": self.total_execution_time / max(self.execution_count, 1)
        }
    
    def _track_execution_time(self, start_time: float):
        """
        Track execution time for metrics.
        
        Args:
            start_time: Start time from time.time()
        """
        execution_time = time.time() - start_time
        self.execution_count += 1
        self.total_execution_time += execution_time
        
        logger.debug(f"Role '{self.name}' execution completed in {execution_time:.3f}s")
    
    def _create_error_result(self, error: Exception, execution_time: float = 0.0) -> Dict[str, Any]:
        """
        Create standardized error result for programmatic execution.
        
        Args:
            error: The exception that occurred
            execution_time: Time spent before error occurred
            
        Returns:
            Dict: Standardized error result
        """
        return {
            "error": str(error),
            "error_type": type(error).__name__,
            "execution_metadata": {
                "role": self.name,
                "execution_type": "programmatic",
                "execution_time": f"{execution_time:.2f}s",
                "success": False
            }
        }
    
    def _create_success_result(self, data: Any, execution_time: float, 
                             llm_calls: int = 0, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create standardized success result for programmatic execution.
        
        Args:
            data: The actual result data
            execution_time: Time spent on execution
            llm_calls: Number of LLM calls made during execution
            metadata: Optional additional metadata
            
        Returns:
            Dict: Standardized success result
        """
        result = {
            "data": data,
            "execution_metadata": {
                "role": self.name,
                "execution_type": "programmatic",
                "execution_time": f"{execution_time:.2f}s",
                "llm_calls": llm_calls,
                "success": True
            }
        }
        
        if metadata:
            result["execution_metadata"].update(metadata)
            
        return result
    
    def __str__(self) -> str:
        """String representation of the programmatic role."""
        return f"ProgrammaticRole(name='{self.name}', executions={self.execution_count})"
    
    def __repr__(self) -> str:
        """Detailed representation of the programmatic role."""
        return (f"ProgrammaticRole(name='{self.name}', description='{self.description}', "
                f"executions={self.execution_count}, avg_time={self.total_execution_time / max(self.execution_count, 1):.3f}s)")