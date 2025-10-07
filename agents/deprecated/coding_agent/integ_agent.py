import unittest
from unittest.mock import Mock
from agents.coding_agent.agent import CodingAgent
from agents.coding_agent.tools.core import draft_code, retrieve_context, use_aider
from agents.coding_agent.types import CodingState
from common.task_graph import TaskPlanOutput
from llm_provider.factory import LLMFactory, LLMType
from common.message_bus import MessageBus
from logging import Logger
from config.bedrock_config import BedrockConfig

class TestSolverAgentIntegration(unittest.TestCase):
    def setUp(self):
        self.logger = Logger(__name__)

        # Configure the LLMFactory
        self.llm_factory = LLMFactory({})
        bedrock_config = BedrockConfig(
            name="bedrock", model='anthropic.claude-3-sonnet-20240229-v1:0'
        )
        self.llm_factory.add_config(LLMType.DEFAULT, bedrock_config)
        self.message_bus = Mock(MessageBus)
        self.agent_id = "solver_agent"
        self.config = {}
        self.agent = CodingAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id, self.config)
        

                
    def test_retrieve_context(self):
        task_data = {
            "prompt": "Update all references to metrics manager update_metrics to include the new timestamp parameter.",
        }

        output: TaskPlanOutput = self.agent.run(task_data)
        
        results = []
        
        state = CodingState(
            prompt=None,
            messages=[],
            inputs={},
            candidate=None,
            node=None,
            llm=self.llm_factory.create_chat_model(LLMType.DEFAULT)
        )

        for task in output.tasks:
            tool = task.tool_id
            
            state["prompt"] = task.prompt

            # Get the class reference dynamically
            cls = globals().get(tool, None)

            if cls is not None:
                # Create an instance of the class
                instance = cls()

                # Call the _run method of the instance
                state = instance._run(state)
                print(state)
                results.append(state)
            else:
                print(f"Tool '{tool}' not found.")
        
        # Arrange

        
        retrieval = retrieve_context()

        # Act
        retrival_state = retrieval._run(state)
        
        docs = [doc.metadata['source'] for doc in retrival_state["context"]]
        
        state["context"] = docs
        
        aider = use_aider()
        
        aider._run(retrival_state)
        
        
        draft = draft_code()
        
        state = draft._run(state)
        
        print(state)

        
        


    def test_run_integration(self):
        test_cases = [
            {"input": "1 2 3 4 5", "output": "6"},
            {"input": "6 7 8 9 10", "output": "18"}
        ]
        runtime_limit = 5  # seconds



 

if __name__ == "__main__":
    unittest.main()
