import unittest
from agents.planning_agent.agent import PlanningAgentOutput, PlanningAgentOutputParser, TaskDescription, TaskDependency

class TestPlanningAgentOutput(unittest.TestCase):
    def test_valid_output(self):
        tasks = [
            TaskDescription(agent_id="agent_1", task_type="fetch_data", prompt="Fetch data from API"),
            TaskDescription(agent_id="agent_1", task_type="process_data", prompt="Process fetched data"),
        ]
        dependencies = [
            TaskDependency(source="fetch_data", target="process_data"),
        ]

        output = PlanningAgentOutput(tasks=tasks, dependencies=dependencies)

        self.assertEqual(len(output.tasks), 2)
        self.assertEqual(len(output.dependencies), 1)

    def test_invalid_task(self):
        tasks = [
            TaskDescription(agent_id="agent_1", task_type="fetch_data", prompt="Fetch data from API"),
            TaskDescription(agent_id=None, task_type=None, prompt=None),
        ]

        with self.assertRaises(ValueError):
            PlanningAgentOutput(tasks=tasks)

    def test_invalid_dependency(self):
        tasks = [
            TaskDescription(agent_id="agent_1", task_type="fetch_data", prompt="Fetch data from API"),
            TaskDescription(agent_id="agent_1", task_type="process_data", prompt="Process fetched data"),
        ]
        dependencies = [
            TaskDependency(source=None, target=None),
        ]

        with self.assertRaises(ValueError):
            PlanningAgentOutput(tasks=tasks, dependencies=dependencies)

class TestPlanningAgentOutputParser(unittest.TestCase):
    def test_valid_output_parsing(self):
        output_text = "Task 1: Fetch data from API\nAgent: agent_1\n\nTask 2: Process data\nAgent: agent_1\n\nDependencies:\nTask 1 -> Task 2"
        parser = PlanningAgentOutputParser()
        planning_agent_output = parser.parse(output_text)

        self.assertEqual(len(planning_agent_output.tasks), 2)
        self.assertEqual(len(planning_agent_output.dependencies), 1)

    def test_invalid_output_parsing(self):
        output_text = "Invalid output format"
        parser = PlanningAgentOutputParser()

        with self.assertRaises(ValueError):
            parser.parse(output_text)


if __name__ == '__main__':
    unittest.main()
