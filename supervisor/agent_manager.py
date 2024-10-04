import importlib
import logging
import os
from typing import Optional, Type, Dict
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

from langchain.agents import AgentType
from supervisor.supervisor_config import SupervisorConfig
from agents.base_agent import BaseAgent
from shared_tools.message_bus import MessageBus
from llm_provider.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

class AgentManager(BaseModel):
    config: SupervisorConfig = Field(..., description="The supervisor configuration")
    message_bus: MessageBus = Field(..., description="The message bus instance")
    agent_registry: Dict[str, AgentType] = Field(default_factory=dict, description="Registry of agents")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, config: SupervisorConfig, message_bus: MessageBus):
        super().__init__(config=config, message_bus=message_bus)

    def register_agent(self, agent: AgentType):
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

    def get_agent(self, agent_id: str) -> Optional[AgentType]:
        return self.agent_registry.get(agent_id)

    def register_agents(self):
        try:
            agents_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'agents'))
            logger.info(f"Searching for agents in directory: {agents_dir}")
            for agent_dir in agents_dir.iterdir():
                if agent_dir.is_dir() and not agent_dir.name.startswith('_'):
                    logger.info(f"Trying to import agent from directory: {agent_dir.name}")
                    try:
                        agent_module = importlib.import_module(f'agents.{agent_dir.name}.agent')
                        agent_classes = [cls for cls in agent_module.__dict__.values() if isinstance(cls, type) and issubclass(cls, BaseAgent) and cls != BaseAgent]
                        if not agent_classes:
                            logger.warning(f"Skipping agent directory '{agent_dir.name}' as it does not contain a valid agent class.")
                            continue
    
                        logger.info(f"Found {len(agent_classes)} agent classes in {agent_dir.name}")
                        for agent_class in agent_classes:
                            try:
                                logger.info(f"Trying to register agent: {agent_class.__name__}")
                                agent_config_file = agent_dir / 'config.yaml'
                                if agent_config_file.exists():
                                    logger.info(f"Loading config from {agent_config_file}")
                                    with open(agent_config_file, 'r') as f:
                                        agent_config = yaml.safe_load(f)
                                else:
                                    logger.info(f"No config file found for {agent_class.__name__}, using default config.")
                                    agent_config = {}
    
                                agent = agent_class(self.config.llm_client_factory, self.message_bus, **agent_config.get("config", {}))
                                self.register_agent(agent)
                                logger.info(f"Successfully registered agent: {agent_class.__name__}")
                            except Exception as e:
                                logger.error(f"Error registering agent '{agent_class.__name__}' from '{agent_dir.name}': {e}")
                    except Exception as e:
                        logger.error(f"Error importing agent from '{agent_dir.name}': {e}")
                else:
                    logger.debug(f"Skipping directory {agent_dir.name} as it is not a directory or starts with an underscore.")
        except Exception as e:
            logger.error(f"Error registering agents: {e}")
