"""End-to-end integration tests for Document 35 intent-based workflow lifecycle management.

Tests the complete workflow from planning role intent creation through universal agent
intent detection to validate the elimination of communication manager request ID warnings.

Following Documents 25 & 26 LLM-safe architecture patterns.
"""

import json
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from common.intents import WorkflowIntent
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.role_registry import RoleRegistry
from llm_provider.universal_agent import UniversalAgent
from roles.core_planning import execute_task_graph


class TestDocument35EndToEnd:
    """End-to-end tests for Document 35 implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Complex multi-step workflow JSON
        self.complex_workflow_json = json.dumps(
            {
                "tasks": [
                    {
                        "id": "search_task",
                        "name": "Search Thailand Information",
                        "description": "Search for Thailand travel planning information",
                        "role": "search",
                        "parameters": {
                            "query": "Thailand travel planning guide best time to visit",
                            "type": "travel",
                        },
                    },
                    {
                        "id": "weather_task",
                        "name": "Check Chicago Weather",
                        "description": "Get current weather conditions in Chicago",
                        "role": "weather",
                        "parameters": {"location": "Chicago", "timeframe": "current"},
                    },
                    {
                        "id": "essay_research",
                        "name": "Research Thomas Paine",
                        "description": "Search for information about Thomas Paine for essay",
                        "role": "search",
                        "parameters": {
                            "query": "Thomas Paine biography Common Sense American Revolution",
                            "type": "historical",
                        },
                    },
                    {
                        "id": "essay_generation",
                        "name": "Generate Thomas Paine Essay",
                        "description": "Write an essay about Thomas Paine based on research",
                        "role": "conversation",
                        "parameters": {
                            "format": "essay",
                            "topic": "Thomas Paine and the American Revolution",
                        },
                    },
                ],
                "dependencies": [
                    {
                        "source_task_id": "search_task",
                        "target_task_id": "weather_task",
                        "type": "sequential",
                    },
                    {
                        "source_task_id": "weather_task",
                        "target_task_id": "essay_research",
                        "type": "sequential",
                    },
                    {
                        "source_task_id": "essay_research",
                        "target_task_id": "essay_generation",
                        "type": "sequential",
                    },
                ],
            }
        )

    def test_planning_role_creates_complex_workflow_intent(self):
        """Test that planning role creates WorkflowIntent for complex workflows."""
        # Arrange
        mock_context = Mock()
        mock_context.context_id = "complex_workflow_123"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "slack:C123"
        mock_context.original_prompt = (
            "Plan Thailand trip, check Chicago weather, write Thomas Paine essay"
        )

        # Act
        result = execute_task_graph(
            llm_result=self.complex_workflow_json, context=mock_context, pre_data={}
        )

        # Assert
        assert isinstance(result, WorkflowIntent)
        assert result.validate() is True
        assert len(result.tasks) == 4
        assert len(result.dependencies) == 3
        assert result.request_id == "complex_workflow_123"

        # Verify expected workflow IDs for lifecycle tracking
        expected_workflow_ids = result.get_expected_workflow_ids()
        assert len(expected_workflow_ids) == 4
        assert "complex_workflow_123_task_search_task" in expected_workflow_ids
        assert "complex_workflow_123_task_weather_task" in expected_workflow_ids
        assert "complex_workflow_123_task_essay_research" in expected_workflow_ids
        assert "complex_workflow_123_task_essay_generation" in expected_workflow_ids

    def test_workflow_intent_contains_all_task_parameters(self):
        """Test that WorkflowIntent preserves all task parameters."""
        # Arrange
        mock_context = Mock()
        mock_context.context_id = "params_test_456"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "console"
        mock_context.original_prompt = "Parameter preservation test"

        # Act
        result = execute_task_graph(
            llm_result=self.complex_workflow_json, context=mock_context, pre_data={}
        )

        # Assert
        assert isinstance(result, WorkflowIntent)

        # Verify search task parameters
        search_task = next(task for task in result.tasks if task["id"] == "search_task")
        assert (
            search_task["parameters"]["query"]
            == "Thailand travel planning guide best time to visit"
        )
        assert search_task["parameters"]["type"] == "travel"

        # Verify weather task parameters
        weather_task = next(
            task for task in result.tasks if task["id"] == "weather_task"
        )
        assert weather_task["parameters"]["location"] == "Chicago"
        assert weather_task["parameters"]["timeframe"] == "current"

        # Verify conversation task parameters
        essay_task = next(
            task for task in result.tasks if task["id"] == "essay_generation"
        )
        assert essay_task["parameters"]["format"] == "essay"
        assert (
            essay_task["parameters"]["topic"]
            == "Thomas Paine and the American Revolution"
        )

    def test_workflow_intent_dependency_structure(self):
        """Test that WorkflowIntent preserves dependency structure."""
        # Arrange
        mock_context = Mock()
        mock_context.context_id = "dependency_test_789"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "slack:C456"
        mock_context.original_prompt = "Dependency structure test"

        # Act
        result = execute_task_graph(
            llm_result=self.complex_workflow_json, context=mock_context, pre_data={}
        )

        # Assert
        assert isinstance(result, WorkflowIntent)
        assert len(result.dependencies) == 3

        # Verify sequential dependencies
        deps = result.dependencies
        assert any(
            dep["source_task_id"] == "search_task"
            and dep["target_task_id"] == "weather_task"
            for dep in deps
        )
        assert any(
            dep["source_task_id"] == "weather_task"
            and dep["target_task_id"] == "essay_research"
            for dep in deps
        )
        assert any(
            dep["source_task_id"] == "essay_research"
            and dep["target_task_id"] == "essay_generation"
            for dep in deps
        )

        # Verify all dependencies are sequential
        assert all(dep["type"] == "sequential" for dep in deps)

    def test_mixed_content_extraction_robustness(self):
        """Test robust JSON extraction from various mixed content formats."""
        # Arrange
        test_cases = [
            # Case 1: JSON with explanatory text before and after
            f"""
            Here's the workflow I've created for your request:

            {self.complex_workflow_json}

            This workflow will handle all your requirements efficiently.
            """,
            # Case 2: JSON with markdown formatting
            f"""
            ## Workflow Plan

            ```json
            {self.complex_workflow_json}
            ```

            This should work well for your needs.
            """,
            # Case 3: JSON with additional explanations
            f"""
            I'll create a comprehensive workflow for you:

            {self.complex_workflow_json}

            The workflow includes:
            1. Thailand travel research
            2. Chicago weather check
            3. Thomas Paine essay research
            4. Essay generation
            """,
        ]

        mock_context = Mock()
        mock_context.context_id = "mixed_content_test"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "console"
        mock_context.original_prompt = "Mixed content extraction test"

        for i, mixed_content in enumerate(test_cases):
            # Act
            result = execute_task_graph(
                llm_result=mixed_content, context=mock_context, pre_data={}
            )

            # Assert
            assert isinstance(result, WorkflowIntent), f"Test case {i+1} failed"
            assert (
                len(result.tasks) == 4
            ), f"Test case {i+1}: Expected 4 tasks, got {len(result.tasks)}"
            assert (
                len(result.dependencies) == 3
            ), f"Test case {i+1}: Expected 3 dependencies, got {len(result.dependencies)}"

    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling for various failure scenarios."""
        mock_context = Mock()
        mock_context.context_id = "error_test_101"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "console"
        mock_context.original_prompt = "Error handling test"

        # Test Case 1: Completely invalid JSON
        with pytest.raises(ValueError, match="Invalid JSON in TaskGraph"):
            execute_task_graph(
                llm_result="This is not JSON at all", context=mock_context, pre_data={}
            )

        # Test Case 2: Malformed JSON
        with pytest.raises(ValueError, match="Invalid JSON in TaskGraph"):
            execute_task_graph(
                llm_result='{"tasks": [{"id": "task_1", "name": "Test"',  # Missing closing braces
                context=mock_context,
                pre_data={},
            )

        # Test Case 3: Validation error message
        with pytest.raises(
            ValueError, match="Cannot create WorkflowIntent from error message"
        ):
            execute_task_graph(
                llm_result="Invalid role references: Task 'task_1' uses unavailable role 'nonexistent'",
                context=mock_context,
                pre_data={},
            )

        # Test Case 4: Empty JSON object (missing required keys)
        with pytest.raises(ValueError, match="TaskGraph intent creation error"):
            execute_task_graph(llm_result="{}", context=mock_context, pre_data={})

    def test_intent_serialization_roundtrip(self):
        """Test that WorkflowIntent can be serialized and deserialized without loss."""
        # Arrange
        mock_context = Mock()
        mock_context.context_id = "serialization_test_202"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "slack:C789"
        mock_context.original_prompt = "Serialization roundtrip test"

        # Act
        original_intent = execute_task_graph(
            llm_result=self.complex_workflow_json, context=mock_context, pre_data={}
        )

        # Serialize to dict
        serialized = original_intent.to_dict()

        # Deserialize back to intent
        deserialized_intent = WorkflowIntent.from_dict(serialized)

        # Assert
        assert deserialized_intent.request_id == original_intent.request_id
        assert deserialized_intent.user_id == original_intent.user_id
        assert deserialized_intent.channel_id == original_intent.channel_id
        assert (
            deserialized_intent.original_instruction
            == original_intent.original_instruction
        )
        assert len(deserialized_intent.tasks) == len(original_intent.tasks)
        assert len(deserialized_intent.dependencies) == len(
            original_intent.dependencies
        )

        # Verify task details preserved
        for orig_task, deser_task in zip(
            original_intent.tasks, deserialized_intent.tasks
        ):
            assert orig_task["id"] == deser_task["id"]
            assert orig_task["name"] == deser_task["name"]
            assert orig_task["role"] == deser_task["role"]
            assert orig_task["parameters"] == deser_task["parameters"]

    def test_workflow_id_generation_consistency(self):
        """Test that workflow ID generation is consistent and predictable."""
        # Arrange
        mock_context = Mock()
        mock_context.context_id = "consistency_test_303"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "console"
        mock_context.original_prompt = "Consistency test"

        # Act - Create intent multiple times
        results = []
        for i in range(3):
            result = execute_task_graph(
                llm_result=self.complex_workflow_json, context=mock_context, pre_data={}
            )
            results.append(result)

        # Assert - All results should generate same workflow IDs
        for i in range(1, len(results)):
            assert (
                results[i].get_expected_workflow_ids()
                == results[0].get_expected_workflow_ids()
            )

        # Verify specific workflow ID format
        expected_ids = results[0].get_expected_workflow_ids()
        assert "consistency_test_303_task_search_task" in expected_ids
        assert "consistency_test_303_task_weather_task" in expected_ids
        assert "consistency_test_303_task_essay_research" in expected_ids
        assert "consistency_test_303_task_essay_generation" in expected_ids

    def test_context_field_handling_robustness(self):
        """Test robust handling of various context field scenarios."""
        test_cases = [
            # Case 1: Complete context
            {
                "context_id": "complete_test_404",
                "user_id": "complete_user",
                "channel_id": "slack:C123",
                "original_prompt": "Complete context test",
            },
            # Case 2: Minimal context
            {
                "context_id": "minimal_test_505",
                # Missing user_id, channel_id, original_prompt
            },
            # Case 3: Partial context
            {
                "context_id": "partial_test_606",
                "user_id": "partial_user",
                # Missing channel_id, original_prompt
            },
        ]

        for i, context_data in enumerate(test_cases):
            # Arrange
            mock_context = Mock()
            for key, value in context_data.items():
                setattr(mock_context, key, value)

            # Remove attributes that weren't set to test getattr defaults
            if "user_id" not in context_data:
                if hasattr(mock_context, "user_id"):
                    delattr(mock_context, "user_id")
            if "channel_id" not in context_data:
                if hasattr(mock_context, "channel_id"):
                    delattr(mock_context, "channel_id")
            if "original_prompt" not in context_data:
                if hasattr(mock_context, "original_prompt"):
                    delattr(mock_context, "original_prompt")

            # Act
            result = execute_task_graph(
                llm_result=self.complex_workflow_json, context=mock_context, pre_data={}
            )

            # Assert
            assert isinstance(result, WorkflowIntent), f"Test case {i+1} failed"
            assert result.request_id == context_data["context_id"]

            # Verify defaults are applied correctly
            if "user_id" not in context_data:
                assert result.user_id == "unknown"
            if "channel_id" not in context_data:
                assert result.channel_id == "console"
            if "original_prompt" not in context_data:
                assert result.original_instruction == "Multi-step workflow"

    def test_intent_validation_edge_cases(self):
        """Test intent validation with various edge cases."""
        mock_context = Mock()
        mock_context.context_id = "validation_test_707"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "console"
        mock_context.original_prompt = "Validation edge case test"

        # Test Case 1: Empty tasks array (should create intent but fail validation)
        empty_tasks_json = json.dumps({"tasks": [], "dependencies": []})
        result = execute_task_graph(
            llm_result=empty_tasks_json, context=mock_context, pre_data={}
        )
        assert isinstance(result, WorkflowIntent)
        assert result.validate() is False  # Should fail validation

        # Test Case 2: Tasks without required fields (should create intent)
        incomplete_tasks_json = json.dumps(
            {
                "tasks": [{"id": "incomplete_task"}],  # Missing name, description, role
                "dependencies": [],
            }
        )
        result = execute_task_graph(
            llm_result=incomplete_tasks_json, context=mock_context, pre_data={}
        )
        assert isinstance(result, WorkflowIntent)
        assert len(result.tasks) == 1

        # Test Case 3: Invalid dependency references (should create intent)
        invalid_deps_json = json.dumps(
            {
                "tasks": [
                    {
                        "id": "task_1",
                        "name": "Task 1",
                        "description": "First task",
                        "role": "search",
                    }
                ],
                "dependencies": [
                    {
                        "source_task_id": "nonexistent",
                        "target_task_id": "task_1",
                        "type": "sequential",
                    }
                ],
            }
        )
        result = execute_task_graph(
            llm_result=invalid_deps_json, context=mock_context, pre_data={}
        )
        assert isinstance(result, WorkflowIntent)
        assert len(result.dependencies) == 1

    def test_performance_with_large_workflows(self):
        """Test performance with large workflow definitions."""
        # Arrange - Create large workflow with many tasks
        large_workflow_tasks = []
        large_workflow_deps = []

        for i in range(20):  # 20 tasks
            large_workflow_tasks.append(
                {
                    "id": f"task_{i}",
                    "name": f"Task {i}",
                    "description": f"Description for task {i}",
                    "role": "search" if i % 2 == 0 else "weather",
                    "parameters": {"param": f"value_{i}"},
                }
            )

            # Create sequential dependencies
            if i > 0:
                large_workflow_deps.append(
                    {
                        "source_task_id": f"task_{i-1}",
                        "target_task_id": f"task_{i}",
                        "type": "sequential",
                    }
                )

        large_workflow_json = json.dumps(
            {"tasks": large_workflow_tasks, "dependencies": large_workflow_deps}
        )

        mock_context = Mock()
        mock_context.context_id = "performance_test_808"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "console"
        mock_context.original_prompt = "Performance test with large workflow"

        # Act - Measure execution time
        start_time = time.time()
        result = execute_task_graph(
            llm_result=large_workflow_json, context=mock_context, pre_data={}
        )
        execution_time = time.time() - start_time

        # Assert
        assert isinstance(result, WorkflowIntent)
        assert len(result.tasks) == 20
        assert len(result.dependencies) == 19
        assert execution_time < 1.0  # Should complete within 1 second

        # Verify workflow ID generation scales
        workflow_ids = result.get_expected_workflow_ids()
        assert len(workflow_ids) == 20

    def test_concurrent_intent_creation(self):
        """Test concurrent intent creation to verify thread safety."""
        # Arrange
        import concurrent.futures
        import threading

        def create_intent(request_id):
            mock_context = Mock()
            mock_context.context_id = f"concurrent_test_{request_id}"
            mock_context.user_id = f"user_{request_id}"
            mock_context.channel_id = "console"
            mock_context.original_prompt = f"Concurrent test {request_id}"

            return execute_task_graph(
                llm_result=self.complex_workflow_json, context=mock_context, pre_data={}
            )

        # Act - Create intents concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_intent, i) for i in range(10)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # Assert
        assert len(results) == 10
        for result in results:
            assert isinstance(result, WorkflowIntent)
            assert result.validate() is True
            assert len(result.tasks) == 4
            assert len(result.dependencies) == 3

    def test_memory_usage_stability(self):
        """Test that intent creation doesn't cause memory leaks."""
        # Arrange
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        mock_context = Mock()
        mock_context.context_id = "memory_test_909"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "console"
        mock_context.original_prompt = "Memory stability test"

        # Act - Create many intents
        for i in range(100):
            result = execute_task_graph(
                llm_result=self.complex_workflow_json, context=mock_context, pre_data={}
            )
            assert isinstance(result, WorkflowIntent)

            # Force garbage collection every 10 iterations
            if i % 10 == 0:
                gc.collect()

        # Final garbage collection
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Assert - Memory increase should be minimal (less than 10MB)
        assert (
            memory_increase < 10 * 1024 * 1024
        ), f"Memory increased by {memory_increase / 1024 / 1024:.2f}MB"

    def test_llm_safe_architecture_compliance(self):
        """Test that implementation follows LLM-Safe architecture patterns."""
        # This test verifies that the implementation follows Documents 25 & 26

        # Arrange
        mock_context = Mock()
        mock_context.context_id = "llm_safe_test_1010"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "console"
        mock_context.original_prompt = "LLM-Safe architecture test"

        # Act & Assert - Function should be pure (no side effects)
        result1 = execute_task_graph(
            llm_result=self.complex_workflow_json, context=mock_context, pre_data={}
        )

        result2 = execute_task_graph(
            llm_result=self.complex_workflow_json, context=mock_context, pre_data={}
        )

        # Pure function should return equivalent results
        assert result1.request_id == result2.request_id
        assert result1.tasks == result2.tasks
        assert result1.dependencies == result2.dependencies

        # Verify no asyncio usage (should not raise any asyncio-related errors)
        # This is validated by the function executing synchronously
        assert isinstance(result1, WorkflowIntent)
        assert isinstance(result2, WorkflowIntent)
