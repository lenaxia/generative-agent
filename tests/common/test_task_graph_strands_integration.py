import json
import unittest

from common.task_graph import TaskDependency, TaskDescription, TaskGraph


class TestTaskGraphStrandsIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for StrandsAgent integration."""
        self.tasks = [
            TaskDescription(
                task_name="planning_task",
                agent_id="planning_agent",
                task_type="planning",
                prompt="Create a comprehensive plan for {input}",
            ),
            TaskDescription(
                task_name="execution_task",
                agent_id="execution_agent",
                task_type="execution",
                prompt="Execute the plan: {input}",
            ),
        ]

        self.dependencies = [
            TaskDependency(
                source="planning_task", target="execution_task", condition=None
            )
        ]

        self.task_graph = TaskGraph(
            tasks=self.tasks,
            dependencies=self.dependencies,
            request_id="test_request_strands",
        )

    def test_task_node_supports_role_based_execution(self):
        """Test that TaskNode can store role information for Universal Agent."""
        task_id = self.task_graph.task_name_map["planning_task"]
        task_node = self.task_graph.get_node_by_task_id(task_id)

        # Set role-based execution info
        task_node.set_role("planning")
        task_node.set_llm_type("STRONG")

        assert task_node.get_role() == "planning"
        assert task_node.get_llm_type() == "STRONG"

    def test_task_node_supports_tool_requirements(self):
        """Test that TaskNode can specify required tools."""
        task_id = self.task_graph.task_name_map["planning_task"]
        task_node = self.task_graph.get_node_by_task_id(task_id)

        # Set required tools
        required_tools = ["create_task_plan", "analyze_dependencies", "web_search"]
        task_node.set_required_tools(required_tools)

        assert task_node.get_required_tools() == required_tools

    def test_task_node_context_passing(self):
        """Test that TaskNode can pass context between nodes."""
        planning_task_id = self.task_graph.task_name_map["planning_task"]
        execution_task_id = self.task_graph.task_name_map["execution_task"]

        planning_node = self.task_graph.get_node_by_task_id(planning_task_id)
        self.task_graph.get_node_by_task_id(execution_task_id)

        # Set context for planning task
        planning_context = {
            "user_request": "Build a web application",
            "constraints": ["budget: $1000", "timeline: 2 weeks"],
            "preferences": ["React", "Node.js"],
        }
        planning_node.set_context(planning_context)

        # Complete planning task with result
        planning_result = {
            "plan": "Step 1: Setup, Step 2: Development, Step 3: Testing",
            "estimated_time": "10 days",
            "resources_needed": ["developer", "designer"],
        }
        self.task_graph.mark_task_completed(
            planning_task_id, json.dumps(planning_result)
        )

        # Get context for execution task (should include planning result)
        execution_context = self.task_graph.get_task_context(execution_task_id)

        assert "parent_results" in execution_context
        assert "conversation_history" in execution_context

        # Verify planning result is available to execution task
        parent_results = execution_context["parent_results"]
        assert len(parent_results) == 1
        assert "plan" in parent_results[0]

    def test_task_graph_conversation_history_integration(self):
        """Test that conversation history is properly integrated with task execution."""
        # Add conversation entries
        self.task_graph.add_conversation_entry("user", "I need to build a web app")
        self.task_graph.add_conversation_entry(
            "assistant", "I'll help you create a plan"
        )

        task_id = self.task_graph.task_name_map["planning_task"]
        context = self.task_graph.get_task_context(task_id)

        assert "conversation_history" in context
        history = context["conversation_history"]
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_task_graph_progressive_summary_integration(self):
        """Test that progressive summary is integrated with task context."""
        # Add progressive summaries
        self.task_graph.add_to_progressive_summary(
            "User requested web application development"
        )
        self.task_graph.add_to_progressive_summary("Planning phase initiated")

        task_id = self.task_graph.task_name_map["execution_task"]
        context = self.task_graph.get_task_context(task_id)

        assert "progressive_summary" in context
        summary = context["progressive_summary"]
        assert len(summary) == 2

    def test_checkpoint_preserves_strands_integration_data(self):
        """Test that checkpoints preserve role-based and tool information."""
        task_id = self.task_graph.task_name_map["planning_task"]
        task_node = self.task_graph.get_node_by_task_id(task_id)

        # Set StrandsAgent-specific data
        task_node.set_role("planning")
        task_node.set_llm_type("STRONG")
        task_node.set_required_tools(["create_plan", "analyze_deps"])
        task_node.set_context({"user_goal": "web app"})

        # Create checkpoint
        checkpoint = self.task_graph.create_checkpoint()

        # Resume from checkpoint
        new_task_graph = TaskGraph(tasks=[], dependencies=[])
        new_task_graph.resume_from_checkpoint(checkpoint)

        # Verify StrandsAgent data is preserved
        restored_node = new_task_graph.get_node_by_task_id(task_id)
        assert restored_node.get_role() == "planning"
        assert restored_node.get_llm_type() == "STRONG"
        assert restored_node.get_required_tools() == ["create_plan", "analyze_deps"]
        assert restored_node.get_context()["user_goal"] == "web app"

    def test_task_execution_with_role_mapping(self):
        """Test that tasks can be executed with proper role mapping."""
        task_id = self.task_graph.task_name_map["planning_task"]
        task_node = self.task_graph.get_node_by_task_id(task_id)

        # Configure for role-based execution
        task_node.set_role("planning")
        task_node.set_llm_type("STRONG")
        task_node.set_required_tools(["create_task_plan"])

        # Simulate task execution preparation
        execution_config = self.task_graph.prepare_task_execution(task_id)

        assert execution_config["role"] == "planning"
        assert execution_config["llm_type"] == "STRONG"
        assert execution_config["tools"] == ["create_task_plan"]
        assert "prompt" in execution_config
        assert "context" in execution_config

    def test_task_result_updates_progressive_summary(self):
        """Test that task completion automatically updates progressive summary."""
        task_id = self.task_graph.task_name_map["planning_task"]

        # Complete task with result
        result = "Created comprehensive project plan with 5 phases"
        self.task_graph.mark_task_completed(task_id, result)

        # Check that progressive summary was updated
        summary = self.task_graph.progressive_summary
        assert len(summary) > 0

        # The last summary entry should reference the completed task
        last_summary = summary[-1]["summary"]
        assert "planning_task" in last_summary.lower()

    def test_parallel_task_execution_context(self):
        """Test context handling for parallel task execution."""
        # Create tasks that can run in parallel
        parallel_tasks = [
            TaskDescription(
                task_name="research_task",
                agent_id="research_agent",
                task_type="research",
                prompt="Research the topic: {input}",
            ),
            TaskDescription(
                task_name="design_task",
                agent_id="design_agent",
                task_type="design",
                prompt="Create design for: {input}",
            ),
        ]

        parallel_graph = TaskGraph(
            tasks=parallel_tasks,
            dependencies=[],  # No dependencies = parallel execution
            request_id="parallel_test",
        )

        # Both tasks should be ready to execute
        ready_tasks = parallel_graph.get_ready_tasks()
        assert len(ready_tasks) == 2

        # Each task should have independent context
        research_id = parallel_graph.task_name_map["research_task"]
        design_id = parallel_graph.task_name_map["design_task"]

        research_context = parallel_graph.get_task_context(research_id)
        design_context = parallel_graph.get_task_context(design_id)

        # Contexts should be independent but share conversation history
        assert (
            research_context["conversation_history"]
            == design_context["conversation_history"]
        )
        assert len(research_context["parent_results"]) == 0
        assert len(design_context["parent_results"]) == 0


if __name__ == "__main__":
    unittest.main()
