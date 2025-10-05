"""
Unit tests for ProgrammaticRole base class.

Tests the abstract base class for programmatic roles that execute directly
without LLM processing, focusing on pure automation and data collection tasks.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional
from abc import ABC

# We'll import this after creating the class
# from llm_provider.programmatic_role import ProgrammaticRole
from common.task_context import TaskContext


class TestProgrammaticRoleBase:
    """Test suite for ProgrammaticRole abstract base class."""
    
    def test_programmatic_role_is_abstract(self):
        """Test that ProgrammaticRole cannot be instantiated directly."""
        # Import after implementation
        from llm_provider.programmatic_role import ProgrammaticRole
        
        with pytest.raises(TypeError):
            ProgrammaticRole("test_role", "Test role description")
    
    def test_programmatic_role_initialization(self):
        """Test proper initialization of ProgrammaticRole subclass."""
        from llm_provider.programmatic_role import ProgrammaticRole
        
        class TestRole(ProgrammaticRole):
            def execute(self, instruction: str, context: Optional[TaskContext] = None) -> Any:
                return {"test": "result"}
            
            def parse_instruction(self, instruction: str) -> Dict[str, Any]:
                return {"query": instruction}
        
        role = TestRole("test_role", "Test role for unit testing")
        
        assert role.name == "test_role"
        assert role.description == "Test role for unit testing"
        assert role.execution_count == 0
        assert role.total_execution_time == 0.0
    
    def test_programmatic_role_execute_abstract_method(self):
        """Test that execute method must be implemented by subclasses."""
        from llm_provider.programmatic_role import ProgrammaticRole
        
        class IncompleteRole(ProgrammaticRole):
            def parse_instruction(self, instruction: str) -> Dict[str, Any]:
                return {"query": instruction}
            # Missing execute method
        
        with pytest.raises(TypeError):
            IncompleteRole("incomplete", "Missing execute method")
    
    def test_programmatic_role_parse_instruction_abstract_method(self):
        """Test that parse_instruction method must be implemented by subclasses."""
        from llm_provider.programmatic_role import ProgrammaticRole
        
        class IncompleteRole(ProgrammaticRole):
            def execute(self, instruction: str, context: Optional[TaskContext] = None) -> Any:
                return {"test": "result"}
            # Missing parse_instruction method
        
        with pytest.raises(TypeError):
            IncompleteRole("incomplete", "Missing parse_instruction method")
    
    def test_programmatic_role_metrics_tracking(self):
        """Test that metrics are properly tracked during execution."""
        from llm_provider.programmatic_role import ProgrammaticRole
        
        class MetricsTestRole(ProgrammaticRole):
            def execute(self, instruction: str, context: Optional[TaskContext] = None) -> Any:
                # Simulate some work - removed sleep for performance
                # time.sleep(0.01)
                self.execution_count += 1
                self.total_execution_time += 0.01
                return {"result": "success"}
            
            def parse_instruction(self, instruction: str) -> Dict[str, Any]:
                return {"query": instruction}
        
        role = MetricsTestRole("metrics_test", "Test metrics tracking")
        
        # Execute multiple times
        role.execute("test instruction 1")
        role.execute("test instruction 2")
        
        metrics = role.get_metrics()
        assert metrics["name"] == "metrics_test"
        assert metrics["execution_count"] == 2
        assert metrics["total_execution_time"] > 0
        assert metrics["average_execution_time"] > 0
    
    def test_programmatic_role_get_metrics(self):
        """Test metrics retrieval functionality."""
        from llm_provider.programmatic_role import ProgrammaticRole
        
        class SimpleRole(ProgrammaticRole):
            def execute(self, instruction: str, context: Optional[TaskContext] = None) -> Any:
                return {"test": "result"}
            
            def parse_instruction(self, instruction: str) -> Dict[str, Any]:
                return {"query": instruction}
        
        role = SimpleRole("simple_role", "Simple test role")
        
        # Test initial metrics
        metrics = role.get_metrics()
        expected_metrics = {
            "name": "simple_role",
            "execution_count": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0
        }
        assert metrics == expected_metrics
        
        # Simulate execution
        role.execution_count = 5
        role.total_execution_time = 2.5
        
        metrics = role.get_metrics()
        assert metrics["execution_count"] == 5
        assert metrics["total_execution_time"] == 2.5
        assert metrics["average_execution_time"] == 0.5
    
    def test_programmatic_role_with_task_context(self):
        """Test programmatic role execution with TaskContext."""
        from llm_provider.programmatic_role import ProgrammaticRole
        
        class ContextAwareRole(ProgrammaticRole):
            def execute(self, instruction: str, context: Optional[TaskContext] = None) -> Any:
                if context:
                    return {
                        "instruction": instruction,
                        "context_available": True,
                        "context_state": str(context.execution_state)
                    }
                return {"instruction": instruction, "context_available": False}
            
            def parse_instruction(self, instruction: str) -> Dict[str, Any]:
                return {"query": instruction}
        
        role = ContextAwareRole("context_role", "Context-aware test role")
        
        # Test without context
        result = role.execute("test instruction")
        assert result["context_available"] is False
        
        # Test with context
        mock_context = Mock(spec=TaskContext)
        mock_context.execution_state = "RUNNING"
        
        result = role.execute("test instruction", context=mock_context)
        assert result["context_available"] is True
        assert result["context_state"] == "RUNNING"
    
    def test_programmatic_role_error_handling(self):
        """Test error handling in programmatic role execution."""
        from llm_provider.programmatic_role import ProgrammaticRole
        
        class ErrorProneRole(ProgrammaticRole):
            def execute(self, instruction: str, context: Optional[TaskContext] = None) -> Any:
                if instruction == "fail":
                    raise ValueError("Simulated execution failure")
                return {"result": "success"}
            
            def parse_instruction(self, instruction: str) -> Dict[str, Any]:
                if instruction == "parse_fail":
                    raise ValueError("Simulated parsing failure")
                return {"query": instruction}
        
        role = ErrorProneRole("error_role", "Error-prone test role")
        
        # Test successful execution
        result = role.execute("success")
        assert result["result"] == "success"
        
        # Test execution error
        with pytest.raises(ValueError, match="Simulated execution failure"):
            role.execute("fail")
        
        # Test parsing error
        with pytest.raises(ValueError, match="Simulated parsing failure"):
            role.parse_instruction("parse_fail")
    
    def test_programmatic_role_return_types(self):
        """Test that programmatic roles can return various data types."""
        from llm_provider.programmatic_role import ProgrammaticRole
        
        class MultiTypeRole(ProgrammaticRole):
            def execute(self, instruction: str, context: Optional[TaskContext] = None) -> Any:
                if instruction == "dict":
                    return {"type": "dictionary", "data": [1, 2, 3]}
                elif instruction == "list":
                    return [{"item": 1}, {"item": 2}]
                elif instruction == "string":
                    return "Simple string result"
                elif instruction == "number":
                    return 42
                else:
                    return None
            
            def parse_instruction(self, instruction: str) -> Dict[str, Any]:
                return {"type": instruction}
        
        role = MultiTypeRole("multi_type", "Multi-type test role")
        
        # Test different return types
        assert isinstance(role.execute("dict"), dict)
        assert isinstance(role.execute("list"), list)
        assert isinstance(role.execute("string"), str)
        assert isinstance(role.execute("number"), int)
        assert role.execute("none") is None
    
    def test_programmatic_role_instruction_parsing(self):
        """Test instruction parsing functionality."""
        from llm_provider.programmatic_role import ProgrammaticRole
        
        class ParsingTestRole(ProgrammaticRole):
            def execute(self, instruction: str, context: Optional[TaskContext] = None) -> Any:
                params = self.parse_instruction(instruction)
                return {"parsed_params": params}
            
            def parse_instruction(self, instruction: str) -> Dict[str, Any]:
                # Simple parsing logic for testing
                words = instruction.split()
                return {
                    "action": words[0] if words else "unknown",
                    "target": words[1] if len(words) > 1 else "default",
                    "word_count": len(words)
                }
        
        role = ParsingTestRole("parsing_test", "Instruction parsing test role")
        
        # Test parsing
        params = role.parse_instruction("search wikipedia")
        assert params["action"] == "search"
        assert params["target"] == "wikipedia"
        assert params["word_count"] == 2
        
        # Test execution with parsing
        result = role.execute("collect data sources")
        parsed = result["parsed_params"]
        assert parsed["action"] == "collect"
        assert parsed["target"] == "data"
        assert parsed["word_count"] == 3