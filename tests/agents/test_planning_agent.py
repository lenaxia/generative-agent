import unittest
from unittest.mock import Mock, patch
from agents.planning_agent.agent import PlanningAgent
from agents.base_agent import BaseAgent
from supervisor.llm_registry import LLMRegistry
from shared_tools.message_bus import MessageBus
from llm_provider.base_client import BaseLLMClient
from langchain.tools import BaseTool
from supervisor.task_graph import TaskGraph
from supervisor.llm_registry import LLMType

class TestPlanningAgent(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.llm_registry = Mock(spec=LLMRegistry)
        self.message_bus = Mock(spec=MessageBus)
        self.agent_id = "planning_agent"
        self.config = {}
        self.planning_agent = PlanningAgent(self.logger, self.llm_registry, self.message_bus, self.agent_id, self.config)

    def test_set_agents(self):
        agents = [Mock(spec=BaseAgent), Mock(spec=BaseAgent)]
        self.planning_agent.set_agents(agents)
        self.assertEqual(self.planning_agent.agents, agents)

    def test_create_tools(self):
        agent1 = Mock(spec=BaseAgent)
        tool1 = Mock(spec=BaseTool, name="tool1")
        tool2 = Mock(spec=BaseTool, name="tool2")
        agent1.tools = {"tool1": tool1, "tool2": tool2}
        agent2 = Mock(spec=BaseAgent)
        tool3 = Mock(spec=BaseTool, name="tool3")
        agent2.tools = {"tool3": tool3}
        self.planning_agent.agents = [agent1, agent2]
        tools = self.planning_agent.create_tools()
        self.assertEqual(len(tools), 3)

    def test_get_tools(self):
        tool1 = Mock(spec=BaseTool, name="tool1")
        tool2 = Mock(spec=BaseTool, name="tool2")
        tool3 = Mock(spec=BaseTool, name="tool3")
        self.planning_agent._tools = {"tool1": tool1, "tool2": tool2, "tool3": tool3}
        tools = self.planning_agent.get_tools()
        self.assertEqual(len(tools), 3)
        self.assertIn("tool1", tools)
        self.assertIn("tool2", tools)
        self.assertIn("tool3", tools)

    @patch("planning_agent.AgentExecutor")
    @patch("planning_agent.parse_obj_as")
    def test_run(self, mock_parse_obj_as, mock_agent_executor):
        llm_client = Mock(spec=BaseLLMClient)
        self.llm_registry.get_client.return_value = llm_client
        instruction = "Test instruction"
        mock_agent = Mock()
        mock_agent_executor.from_tools_and_prompt.return_value = mock_agent
        mock_agent.return_value = {"task_graph": "sample_task_graph"}
        mock_parse_obj_as.return_value = TaskGraph([])

        task_graph = self.planning_agent._run(llm_client, instruction, LLMType.DEFAULT)

        self.llm_registry.get_client.assert_called_once_with(LLMType.DEFAULT)
        self.planning_agent._format_input.assert_called_once_with(instruction)
        mock_agent_executor.from_tools_and_prompt.assert_called_once()
        mock_agent.assert_called_once()
        mock_parse_obj_as.assert_called_once_with(TaskGraph, {"task_graph": "sample_task_graph"})
        self.assertIsInstance(task_graph, TaskGraph)

    def test_format_input(self):
        instruction = "Test instruction"
        formatted_input = self.planning_agent._format_input(instruction)
        self.assertEqual(formatted_input, instruction)

    def test_process_output(self):
        task_graph = TaskGraph([])
        processed_output = self.planning_agent._process_output(task_graph)
        self.assertEqual(processed_output, task_graph)

if __name__ == '__main__':
    unittest.main()
