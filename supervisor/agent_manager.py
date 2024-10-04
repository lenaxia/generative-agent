import importlib
import logging
import os
from typing import Optional, Type
from pathlib import Path
import yaml

from langchain.agents import AgentType
from config import SupervisorConfig
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self, config: SupervisorConfig, message_bus: MessageBus):
        self.config = config
        self.message_bus = message_bus
        self.agent_registry: Dict[str, AgentType] = {}

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
            for agent_dir in agents_dir.iterdir():
                if agent_dir.is_dir() and not agent_dir.name.startswith('_'):
                    agent_module = importlib.import_module(f'agents.{agent_dir.name}.agent')
                    agent_classes = [cls for cls in agent_module.__dict__.values() if isinstance(cls, type) and issubclass(cls, BaseAgent) and cls != BaseAgent]
                    if not agent_classes:
                        logger.warning(f"Skipping agent directory '{agent_dir.name}' as it does not contain a valid agent class.")
                        continue

                    for agent_class in agent_classes:
                        try:
                            agent_config_file = agent_dir / 'config.yaml'
                            if agent_config_file.exists():
                                with open(agent_config_file, 'r') as f:
                                    agent_config = yaml.safe_load(f)
                            else:
                                agent_config = {}

                            agent = agent_class(self.config.llm_client_factory, self.message_bus, **agent_config.get("config", {}))
                            self.register_agent(agent)
                        except Exception as e:
                            logger.error(f"Error registering agent '{agent_class.__name__}' from '{agent_dir.name}': {e}")
        except Exception as e:
            logger.error(f"Error registering agents: {e}")
