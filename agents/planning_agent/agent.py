import logging
from logging import Logger
from typing import List, Dict, Optional
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from pydantic import parse_obj_as
from agents.base_agent import BaseAgent
from shared_tools.message_bus import MessageBus
from supervisor.llm_registry import LLMRegistry, LLMType
from supervisor.task_graph import TaskGraph
from llm_provider.base_client import BaseLLMClient

class PlanningAgent(BaseAgent):
    def __init__(self, logger: Logger, llm_registry: LLMRegistry, message_bus: MessageBus, agent_id: str, config: Optional[Dict] = None):
        super().__init__(logger, llm_registry, message_bus, agent_id, config)
        self.agents = []
        self._tools = None  # Initialize tools as None
        self.logger.info(f"Initialized PlanningAgent with ID: {agent_id}")

    def set_agents(self, agents: List[BaseAgent]):
        """
        Sets the list of available agents for the PlanningAgent.
        """
        self.agents = agents
        self.logger.info(f"Set agents for PlanningAgent: {[agent.__class__.__name__ for agent in agents]}")

    def create_tools(self) -> List[BaseTool]:
        self.logger.debug("Creating tools for PlanningAgent...")
        tools = []
        for agent in self.agents:
            agent_tools = agent.tools.values()  # Get the tool instances directly
            tools.extend(agent_tools)  # Add the tool instances to the tools list
            self.logger.debug(f"Added tools: {[tool.name for tool in agent_tools]}")
        self.logger.info(f"Created {len(tools)} tools for PlanningAgent")
        return tools

    def get_tools(self) -> Dict[str, BaseTool]:
        if self._tools is None:
            self._tools = {tool.name: tool for tool in self.create_tools()}
        return self._tools

    def _run(self, llm_client: BaseLLMClient, instruction: str, llm_type: LLMType = LLMType.DEFAULT) -> TaskGraph:
        self.logger.info(f"Running PlanningAgent with instruction: {instruction}")
        prompt_template = PromptTemplate(
            input_variables=["input", "agents"],
            template="You are a planning agent responsible for breaking down complex tasks into a sequence of smaller subtasks. "
                     "Given the available agents and their capabilities, create a task graph to accomplish the following task: {input}\n\n"
                     "Agents:\n{agents}\n\n"
                     "Task Graph:",
            formatter={
                "agents": lambda agents: "\n".join([f"- {agent.__class__.__name__} ({agent.description})\n  Tools:\n    {' '.join([f'- {tool.name} ({tool.description})' for tool_name, tool in agent.tools.items()])}" for agent in agents])
            },
        )

        self.logger.info(f"Prompt template created.")
        tools = self.get_tools().values()
        self.logger.info(f"Tools for PlanningAgent: {[tool.name for tool in tools]}")

        agent = AgentExecutor.from_tools_and_prompt(
            tools=list(tools),
            llm=llm_client.model,
            prompt=prompt_template,
        )

        self.logger.info("Executing PlanningAgent...")
        response = agent({"input": instruction, "agents": self.agents}, return_intermediate_steps=True)
        self.logger.debug(f"PlanningAgent response: {response}")

        task_graph = parse_obj_as(TaskGraph, response)
        self.logger.info(f"PlanningAgent generated task graph: {task_graph}")
        return task_graph

    def _format_input(self, instruction: str, *args, **kwargs) -> str:
        self.logger.info(f"Formatting input for PlanningAgent: {instruction}")
        return instruction

    def _process_output(self, task_graph: TaskGraph) -> TaskGraph:
        self.logger.info(f"Processing output for PlanningAgent: {task_graph}")
        return task_graph
