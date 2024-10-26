import json
from langchain.output_parsers import OutputFixingParser
from typing import List, Optional, Dict, Any
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, field_validator, ValidationError
from uuid import uuid4
from enum import Enum
from common.message_bus import MessageBus
from llm_provider.factory import LLMFactory, LLMType
from agents.base_agent import BaseAgent, AgentInput
from common.task_graph import TaskGraph, TaskDescription, TaskDependency

class PlanningAgentInput(AgentInput):
    request_id: str

class PlanningAgentOutput(BaseModel):
    tasks: List[TaskDescription] = None
    dependencies: Optional[List[TaskDependency]] = None

    @field_validator('tasks')
    def check_tasks(cls, tasks):
        for task in tasks:
            if not task.task_name or not task.agent_id or not task.task_type or not task.prompt:
                raise ValueError("All tasks must have agent_id, task_type, and prompt.")
        return tasks

    @field_validator('dependencies')
    def check_dependencies(cls, dependencies):
        if dependencies is None or len(dependencies) == 0:
            return dependencies  # Return as is if dependencies is None or empty

        for dependency in dependencies:
            if not dependency.source or not dependency.target:
                raise ValueError("All dependencies must have source and target set.")
        return dependencies

class PlanningAgent(BaseAgent):
    def __init__(self, logger, llm_factory, message_bus, agent_id, config=None):
        super().__init__(logger, llm_factory, message_bus, agent_id, config)
        self.agent_manager = None
        self.logger.info(f"Initialized PlanningAgent with ID: {agent_id}")
        self.feedback_added = False
        self.agent_description = "Generate a plan for completing the given task"

    def set_agent_manager(self, agent_manager):
        self.agent_manager = agent_manager
        self.logger.info(f"Set AgentManager for PlanningAgent")

    def _select_llm_provider(self, llm_type, **kwargs):
        llm = self.llm_factory.create_chat_model(llm_type)
        return llm

    def _run(self, input: PlanningAgentInput):
        self.logger.info(f"Running PlanningAgent with instruction: {input.prompt}")
        self.request_id = input.request_id

        parser = PydanticOutputParser(pydantic_object=PlanningAgentOutput)

        prompt_template = PromptTemplate(
            input_variables=["input", "agents"],
            template="You are a planning agent responsible for breaking down complex tasks into a sequence of smaller subtasks. "
                     "Only use the SearchAgent when a more specialized agent is not available or you lack the knowledge necessary to complete the task."
                     "Given the available agents and their capabilities, create a task graph to accomplish the following task: {input}\n\n"
                     "Agents:\n{agents}\n\n"
                     "Your output should follow the provided JSON schema:\n\n{format_instructions}\n\n"
                     "Only respond with the JSON and no additional formatting or comments, do not include anything else besides the json output\n"
                     "Task Graph:\n",
            #formatter={"agents": lambda agents: "\n".join([f"- {agent}" for agent in agents])},
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        #self.logger.info({"agents": lambda agents: "\n".join([f"- {agent}" for agent in agents])})

        # TODO: Instead of just getting the agent names, also get the agent descriptions, so the LLM has more context on what each agent does
        #   and update the formatter above to properly format the agent name and description
        agents = [agent for agent in self.agent_manager.get_agents()]
        agents_prompt =  "\n".join([f"- {agent.agent_id} ({agent.agent_description})" for agent in agents])
        self.logger.info(f"Agents for PlanningAgent: {agents_prompt}")
        
        llm_provider = self._select_llm_provider(input.llm_type)

        chain = prompt_template | llm_provider | parser
        planning_agent_output = chain.invoke({"input": input.prompt, "agents": agents_prompt})

        self.logger.info(f"PlanningAgent generated task graph: {planning_agent_output}")

        return planning_agent_output


    def _format_input(self, task_data: Dict) -> AgentInput:
        """
        Formats the input for the LLM provider and the respective tool(s).
        """
        input: AgentInput = PlanningAgentInput(**task_data)
        return input

    def _process_output(self, task_data: Dict, output: PlanningAgentOutput) -> TaskGraph:
        tasks = None
        dependencies = None
        request_id = task_data.get("request_id", "undefined")

        try:
            tasks = output.tasks
            dependencies = output.dependencies
        except Exception as e:
            print(e)

        task_graph = TaskGraph(tasks=tasks, dependencies=dependencies, request_id=request_id)
        return task_graph

    def run(self, task_data: Dict):
        self.setup()
        input = self._format_input(task_data)
        planning_agent_output = self._run(input)
        self.teardown()
        return self._process_output(task_data, planning_agent_output)

    def setup(self):
        pass

    def teardown(self):
        pass
