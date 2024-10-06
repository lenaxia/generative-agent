import unittest
from pydantic import parse_obj_as
from unittest.mock import Mock, patch
from agents.planning_agent.agent import PlanningAgent, PlanningAgentOutput, TaskDescription, TaskDependency
from llm_provider.factory import LLMFactory, LLMType
from shared_tools.message_bus import MessageBus
from supervisor.agent_manager import AgentManager
from supervisor.supervisor_config import SupervisorConfig
from config.bedrock_config import BedrockConfig
from langchain_core.runnables.base import Runnable

class TestPlanningAgent(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.message_bus = MessageBus()
        self.supervisor_config = SupervisorConfig()
        self.llm_factory = LLMFactory({})

        # Configure the LLM provider
        bedrock_config = BedrockConfig(
            name="bedrock", model_id='anthropic.claude-3-sonnet-20240229-v1:0', model_kwargs={'temperature': 0}
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
                TaskDescription(task_name="fetch_data", agent_id="agent_1", task_type="fetch_data", prompt_template="Fetch data from API"),
                TaskDescription(task_name="process_data", agent_id="agent_1", task_type="process_data", prompt_template="Process fetched data"),
            ],
            dependencies=[
                TaskDependency(source="fetch_data", target="process_data"),
            ],
        )
    
        mock_invoke.return_value = mock_output
    
        instruction = "Fetch and process data from an API"
        task_graph = self.planning_agent.run(instruction)
    
        self.assertIsNotNone(task_graph)
        self.assertEqual(len(task_graph.nodes), 2)
        self.assertEqual(len(task_graph.edges), 1)

    def test_format_input(self):
        instruction = "Fetch weather data for Seattle"
        formatted_input = self.planning_agent._format_input(instruction)
        self.assertEqual(formatted_input, instruction)


    @patch('pydantic.parse_obj_as')
    def test_process_output(self, mock_parse_obj_as):
        mock_output = PlanningAgentOutput(
            tasks=[
                TaskDescription(task_name="fetch_data", agent_id="agent_1", task_type="fetch_data", prompt_template="Fetch data from API"),
                TaskDescription(task_name="process_data", agent_id="agent_1", task_type="process_data", prompt_template="Process fetched data"),
            ],
            dependencies=[
                TaskDependency(source="fetch_data", target="process_data"),
            ],
        )
        mock_parse_obj_as.return_value = mock_output
    
        output = "Task 1: Fetch data from API\nAgent: agent_1\n\nTask 2: Process data\nAgent: agent_1\n\nDependencies:\nTask 1 -> Task 2"
        instruction = "Fetch and process data from an API"
        task_graph = self.planning_agent._process_output(mock_output, instruction)
    
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
