import unittest
from pydantic import parse_obj_as
from unittest.mock import Mock, patch
from agents.base_agent import AgentInput
from agents.planning_agent.agent import PlanningAgent, PlanningAgentInput, PlanningAgentOutput, TaskDescription, TaskDependency
from llm_provider.factory import LLMFactory, LLMType
from shared_tools.message_bus import MessageBus
from supervisor.agent_manager import AgentManager
from supervisor.supervisor_config import LoggingConfig, SupervisorConfig
from config.bedrock_config import BedrockConfig
from langchain_core.runnables.base import Runnable
from typing import Dict

class TestPlanningAgent(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.message_bus = MessageBus()
        self.supervisor_config = SupervisorConfig(logging=LoggingConfig(log_level="INFO"))
        self.llm_factory = LLMFactory({})

        # Configure the LLM provider
        bedrock_config = BedrockConfig(
            name="bedrock", model='anthropic.claude-3-sonnet-20240229-v1:0'
        )
        self.llm_factory.add_config(LLMType.DEFAULT, bedrock_config)

        # Create the AgentManager
        self.agent_manager = AgentManager(self.supervisor_config, self.message_bus, self.llm_factory)
        self.agent_manager.register_agents()

        self.agent_id = "planning_agent"
        self.planning_agent = PlanningAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id)
        self.planning_agent.set_agent_manager(self.agent_manager)

    @patch('langchain_core.runnables.base.Runnable.invoke')
    def test_run(self, mock_invoke):
        mock_output = PlanningAgentOutput(
            tasks=[
                TaskDescription(task_name="fetch_data", agent_id="agent_1", task_type="fetch_data", prompt="Fetch data from API"),
                TaskDescription(task_name="process_data", agent_id="agent_1", task_type="process_data", prompt="Process fetched data"),
            ],
            dependencies=[
                TaskDependency(source="fetch_data", target="process_data"),
            ],
        )
    
        mock_invoke.return_value = mock_output
    
        input: Dict = dict(
            task_id="task_1",
            task_name="fetch_data",
            request_id="request_1",
            agent_id="agent_1",
            task_type="fetch_data",
            prompt="Fetch data from API",
            status="pending",
            inbound_edges=[],
            outbound_edges=[],
            include_full_history=False,
        )
        task_graph = self.planning_agent.run(input)
    
        self.assertIsNotNone(task_graph)
        self.assertGreaterEqual(len(task_graph.nodes), 2)
        self.assertGreaterEqual(len(task_graph.edges), 1)

    def test_format_input(self):
        prompt = "Fetch weather data for Seattle"
        history = ["This is some maybe relevant history", "This is some more history"]
        formatted_input = self.planning_agent._format_input({"prompt": prompt, "history": history, "request_id": "request_1"})
        expected_input = PlanningAgentInput(prompt=prompt, history=history, request_id="request_1")
        self.assertEqual(formatted_input, expected_input)


    @patch('pydantic.parse_obj_as')
    def test_process_output(self, mock_parse_obj_as):
        mock_output = PlanningAgentOutput(
            tasks=[
                TaskDescription(task_name="fetch_data", agent_id="agent_1", task_type="fetch_data", prompt="Fetch data from API"),
                TaskDescription(task_name="process_data", agent_id="agent_1", task_type="process_data", prompt="Process fetched data"),
            ],
            dependencies=[
                TaskDependency(source="fetch_data", target="process_data"),
            ],
        )
        mock_parse_obj_as.return_value = mock_output
        
        input: Dict = dict(
            task_id="task_1",
            task_name="fetch_data",
            request_id="request_1",
            agent_id="agent_1",
            task_type="fetch_data",
            prompt="Fetch data from API",
            status="pending",
            inbound_edges=[],
            outbound_edges=[],
            include_full_history=False,
        )
    
        output = "Task 1: Fetch data from API\nAgent: agent_1\n\nTask 2: Process data\nAgent: agent_1\n\nDependencies:\nTask 1 -> Task 2"
        instruction = "Fetch and process data from an API"
        task_graph = self.planning_agent._process_output(input, mock_output)
    
        self.assertIsNotNone(task_graph)
        self.assertEqual(len(task_graph.nodes), 2)
        self.assertEqual(len(task_graph.edges), 1)

    def test_setup(self):
        self.planning_agent.setup()
        # No assertions needed as the setup method is currently empty

    def test_teardown(self):
        self.planning_agent.teardown()
        # No assertions needed as the teardown method is currently empty

if __name__ == "__main__":
    unittest.main()
