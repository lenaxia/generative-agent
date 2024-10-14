import importlib
import logging
import os
import sys
from typing import Optional, Type, Dict
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

from langchain.agents import AgentType
from supervisor.supervisor_config import SupervisorConfig
from agents.base_agent import BaseAgent
from shared_tools.message_bus import MessageBus
from llm_provider.factory import LLMFactory, LLMType
from agents.summarizer_agent.agent import TextSummarizerAgent
from agents.search_agent.agent import SearchAgent
from agents.weather_agent.agent import WeatherAgent
from agents.planning_agent.agent import PlanningAgent

logger = logging.getLogger(__name__)

class AgentManager(BaseModel):
    config: SupervisorConfig = Field(..., description="The supervisor configuration")
    message_bus: MessageBus = Field(..., description="The message bus instance")
    llm_factory: LLMFactory = Field(..., description="The LLM factory instance")
    agent_registry: Dict[str, BaseAgent] = Field(default_factory=dict, description="Registry of agents")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, config: SupervisorConfig, message_bus: MessageBus, llm_factory: LLMFactory):
        super().__init__(config=config, message_bus=message_bus, llm_factory=llm_factory)

    def register_agent(self, agent: BaseAgent):
        try:
            self.agent_registry[agent.agent_id] = agent
            logger.info(f"Registered agent '{agent.agent_id}'.")
        except Exception as e:
            logger.error(f"Error registering agent '{agent.agent_id}': {e}")

    def unregister_agent(self, agent_id: str):
        try:
            if agent_id in self.agent_registry:
                del self.agent_registry[agent_id]
                logger.info(f"Unregistered agent '{agent_id}'.")
        except Exception as e:
            logger.error(f"Error unregistering agent '{agent_id}': {e}")

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        return self.agent_registry.get(agent_id)

    def get_agents(self) -> list[BaseAgent]:
        return list(self.agent_registry.values())

    def register_agents(self):
        """
        Registers agents specified in the configuration file.

        The configuration file should specify the list of agents to register.
        For each agent, the configuration file should specify the name of the agent,
        the class of the agent, and the path to the configuration file for the agent.

        The configuration file is expected to be in yaml format.

        If the agent configuration file does not exist, the agent will be registered
        with the default configuration.

        If there is an error registering an agent, the error will be logged and
        the agent will not be registered.

        :return: A list of registered agents.
        """
        # TODO: Move to using dynamic loading of agents using importlib (see commented out code below). 
        try:
            agents_to_register = [
                {
                    'name': 'TextSummarizerAgent',
                    'class': TextSummarizerAgent,
                    'config_path': 'agents/summarizer_agent/config.yaml'
                },
                {
                    'name': 'SearchAgent',
                    'class': SearchAgent,
                    'config_path': 'agents/search_agent/config.yaml'
                },
                {
                    'name': 'PlanningAgent',
                    'class': PlanningAgent,
                    'config_path': 'agents/planning_agent/config.yaml'
                },
                {
                    'name': 'WeatherAgent',
                    'class': WeatherAgent,
                    'config_path': 'agents/weather_agent/config.yaml'
                },
            ]
    
            registered_agents = []
    
            for agent in agents_to_register:
                logger.info(f"Trying to register agent: {agent['name']}")
                agent_config = {}
    
                agent_config_file = Path(__file__).parent.parent / agent['config_path']
                if os.path.exists(agent_config_file):
                    logger.info(f"Loading config from {agent_config_file}")
                    with open(agent_config_file, 'r') as f:
                        agent_config = yaml.safe_load(f)
                else:
                    logger.warning(f"No config file found for {agent['name']}, using default config.")
    
                try:
                    agent_instance = agent['class'](logger, self.llm_factory, self.message_bus, config=agent_config.get("config", {}), agent_id=agent['name'])
                    self.register_agent(agent_instance)
                    registered_agents.append(agent_instance)
                    logger.info(f"Successfully registered agent: {agent['name']}")
                except Exception as e:
                    logger.error(f"Error registering agent '{agent['name']}': {e}")
                    continue
    
        except Exception as e:
            logger.error(f"Error registering agents: {e}")


# TODO: Get the below code working. Right now the problem is that when using importlib, the agent module has a new context outside of the primary project,
#       so it cannot find some of the other modules, like LLMFactory, MessageBus, etc.

#    def register_agents(self):
#        try:
#            agents_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'agents'))
#            logger.info(f"Searching for agents in directory: {agents_dir}")
#    
#            # Add the project root to the Python path
#            project_root = os.path.join(agents_dir.parent.resolve())
#            sys.path.append(project_root)
#    
#            # Add the subdirectories to the Python path
#            subdirs = [os.path.join(project_root, d) for d in os.listdir(project_root) if os.path.isdir(os.path.join(project_root, d))]
#            for subdir in subdirs:
#                sys.path.append(subdir)
#    
#            for agent_dir in agents_dir.iterdir():
#                if agent_dir.is_dir() and not agent_dir.name.startswith('_'):
#                    logger.info(f"Trying to import agent from directory: {agent_dir.name}")
#                    try:
#                        agent_module = importlib.import_module(f'agents.{agent_dir.name}.agent')
#                        agent_classes = [cls for cls in agent_module.__dict__.values() if isinstance(cls, type) and issubclass(cls, BaseAgent) and cls != BaseAgent]
#                        if not agent_classes:
#                            logger.warning(f"Skipping agent directory '{agent_dir.name}' as it does not contain a valid agent class.")
#                            continue
#    
#                        logger.info(f"Found {len(agent_classes)} agent classes in {agent_dir.name}")
#                        for agent_class in agent_classes:
#                            try:
#                                logger.info(f"Trying to register agent: {agent_class.__name__}")
#                                agent_config_file = agent_dir / 'config.yaml'
#                                if agent_config_file.exists():
#                                    logger.info(f"Loading config from {agent_config_file}")
#                                    with open(agent_config_file, 'r') as f:
#                                        agent_config = yaml.safe_load(f)
#                                else:
#                                    logger.info(f"No config file found for {agent_class.__name__}, using default config.")
#                                    agent_config = {}
#    
#                                agent = agent_class(self.config.llm_client_factory, self.message_bus, **agent_config.get("config", {}))
#                                self.register_agent(agent)
#                                logger.info(f"Successfully registered agent: {agent_class.__name__}")
#                            except Exception as e:
#                                logger.error(f"Error registering agent '{agent_class.__name__}' from '{agent_dir.name}': {e}")
#                    except Exception as e:
#                        logger.error(f"Error importing agent from '{agent_dir.name}': {e}")
#                else:
#                    logger.debug(f"Skipping directory {agent_dir.name} as it is not a directory or starts with an underscore.")
#        except Exception as e:
#            logger.error(f"Error registering agents: {e}")
