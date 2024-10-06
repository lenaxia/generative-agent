import json
from langchain.output_parsers import OutputFixingParser
from typing import List
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, validator, ValidationError
from pydantic import parse_obj_as
from uuid import uuid4
from enum import Enum
from shared_tools.message_bus import MessageBus
from llm_provider.factory import LLMFactory, LLMType
from agents.base_agent import BaseAgent
from supervisor.task_graph import TaskGraph, TaskDescription, TaskDependency

class PlanningAgentOutput(BaseModel):
    tasks: List[TaskDescription]
    dependencies: List[TaskDependency] = None

    @validator('tasks')
    def check_tasks(cls, tasks):
        for task in tasks:
            if not task.task_name or not task.agent_id or not task.task_type or not task.prompt_template:
                raise ValueError("All tasks must have agent_id, task_type, and prompt_template set.")
        return tasks

    @validator('dependencies', each_item=True)
    def check_dependencies(cls, dependency):
        if not dependency.source or not dependency.target:
            raise ValueError("All dependencies must have source and target set.")
        return dependency

class PlanningAgent(BaseAgent):
    def __init__(self, logger, llm_factory, message_bus, agent_id, config=None):
        super().__init__(logger, llm_factory, message_bus, agent_id, config)
        self.agent_manager = None
        self.logger.info(f"Initialized PlanningAgent with ID: {agent_id}")
        self.feedback_added = False

    def set_agent_manager(self, agent_manager):
        self.agent_manager = agent_manager
        self.logger.info(f"Set AgentManager for PlanningAgent: {agent_manager}")

    def _select_llm_provider(self, llm_type, **kwargs):
        llm = self.llm_factory.create_chat_model(llm_type)
        return llm

    def _run(self, instruction, llm_type=LLMType.DEFAULT):
        self.logger.info(f"Running PlanningAgent with instruction: {instruction}")

        parser = PydanticOutputParser(pydantic_object=PlanningAgentOutput)

        prompt_template = PromptTemplate(
            input_variables=["input", "agents"],
            template="You are a planning agent responsible for breaking down complex tasks into a sequence of smaller subtasks. "
                     "Given the available agents and their capabilities, create a task graph to accomplish the following task: {input}\n\n"
                     "Agents:\n{agents}\n\n"
                     "Your output should follow the provided JSON schema:\n\n{format_instructions}\n\n"
                     "Only respond with the JSON and no additional formatting or comments, do not include anything else besides the json output\n"
                     "Task Graph:\n",
            formatter={"agents": lambda agents: "\n".join([f"- {agent.agent_id} ({agent.description})" for agent in agents])},
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        agents = self.agent_manager.get_agents()
        self.logger.info(f"Agents for PlanningAgent: {[agent.agent_id for agent in agents]}")

        llm_provider = self._select_llm_provider(llm_type)

        chain = prompt_template | llm_provider | parser
        planning_agent_output = chain.invoke({"input": instruction, "agents": agents})

        self.logger.info(f"PlanningAgent generated task graph: {planning_agent_output}")

        return planning_agent_output


    def _format_input(self, instruction, *args, **kwargs):
        self.logger.info(f"Formatting input for PlanningAgent: {instruction}")
        return instruction

    def _process_output(self, output, instruction):
        self.logger.info(f"Processing output for PlanningAgent: {output}")
        tasks = None
        dependencies = None

        try:
            tasks = output.tasks
            dependencies = output.dependencies
        except Exception as e:
            print(e)

        task_graph = TaskGraph(tasks=tasks, dependencies=dependencies)
        return task_graph

    def run(self, instruction, llm_type=LLMType.DEFAULT, *args, **kwargs):
        self.setup()
        planning_agent_output = self._run(instruction, llm_type)
        self.teardown()
        return self._process_output(planning_agent_output, instruction)

    def setup(self):
        pass

    def teardown(self):
        pass
