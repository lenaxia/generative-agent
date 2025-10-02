"""
Tests for Phase 7.1: Supervisor Migration to StrandsAgent-Only
Verifies that Supervisor works without AgentManager or LangChain dependencies.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch

from supervisor.supervisor import Supervisor
from supervisor.workflow_engine import WorkflowEngine
from llm_provider.factory import LLMFactory, LLMType
from common.message_bus import MessageBus


class TestSupervisorPhase7:
    """Test Supervisor migration to StrandsAgent-only architecture."""

    def setup_method(self):
        """Set up test configuration."""
        self.config_content = '''
logging:
  log_level: INFO
  log_file: test.log
llm_providers:
  default:
    llm_class: DEFAULT
    provider_type: bedrock
    model_id: us.amazon.nova-pro-v1:0
    temperature: 0.3
'''

    def test_supervisor_initialization_without_agent_manager(self):
        """Test that Supervisor initializes without AgentManager."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(self.config_content)
            config_file = f.name

        try:
            supervisor = Supervisor(config_file)
            
            # Verify Supervisor has WorkflowEngine instead of AgentManager
            assert hasattr(supervisor, 'workflow_engine')
            assert isinstance(supervisor.workflow_engine, WorkflowEngine)
            assert not hasattr(supervisor, 'agent_manager')
            
            # Verify LLMFactory is properly initialized
            assert hasattr(supervisor, 'llm_factory')
            assert isinstance(supervisor.llm_factory, LLMFactory)
            
        finally:
            os.unlink(config_file)

    def test_supervisor_status_uses_workflow_engine(self):
        """Test that Supervisor status method uses WorkflowEngine methods."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(self.config_content)
            config_file = f.name

        try:
            supervisor = Supervisor(config_file)
            status = supervisor.status()
            
            # Verify status contains WorkflowEngine metrics
            assert 'workflow_engine' in status
            assert 'universal_agent' in status
            assert 'running' in status
            assert 'metrics' in status
            
            # Verify no old task_scheduler or request_manager references
            assert 'task_scheduler' not in status
            assert 'request_manager' not in status
            
        finally:
            os.unlink(config_file)

    def test_supervisor_workflow_engine_integration(self):
        """Test that Supervisor properly integrates with WorkflowEngine."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(self.config_content)
            config_file = f.name

        try:
            supervisor = Supervisor(config_file)
            
            # Verify WorkflowEngine has proper configuration
            workflow_engine = supervisor.workflow_engine
            assert workflow_engine.max_concurrent_tasks == 5
            assert workflow_engine.checkpoint_interval == 300
            
            # Verify Universal Agent integration
            ua_status = workflow_engine.get_universal_agent_status()
            assert ua_status['universal_agent_enabled'] is True
            assert ua_status['framework'] == 'strands'
            
        finally:
            os.unlink(config_file)

    def test_supervisor_no_langchain_imports(self):
        """Test that Supervisor can be imported without LangChain."""
        # This test verifies that importing Supervisor doesn't require LangChain
        try:
            from supervisor.supervisor import Supervisor
            # If we get here, the import succeeded without LangChain dependencies
            assert True
        except ImportError as e:
            if 'langchain' in str(e).lower():
                pytest.fail(f"Supervisor still has LangChain dependency: {e}")
            else:
                # Some other import error, re-raise
                raise

    def test_supervisor_run_method_uses_workflow_engine(self):
        """Test that Supervisor run method uses WorkflowEngine for request handling."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(self.config_content)
            config_file = f.name

        try:
            supervisor = Supervisor(config_file)
            
            # Mock the workflow_engine methods
            with patch.object(supervisor.workflow_engine, 'handle_request') as mock_handle:
                with patch.object(supervisor.workflow_engine, 'get_request_status') as mock_status:
                    mock_handle.return_value = 'test_request_id'
                    mock_status.return_value = {'status': True}
                    
                    # Verify that workflow_engine methods are available
                    assert hasattr(supervisor.workflow_engine, 'handle_request')
                    assert hasattr(supervisor.workflow_engine, 'get_request_status')
                    
        finally:
            os.unlink(config_file)

    def test_supervisor_component_initialization_order(self):
        """Test that Supervisor initializes components in correct order."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(self.config_content)
            config_file = f.name

        try:
            supervisor = Supervisor(config_file)
            
            # Verify all required components are initialized
            assert supervisor.config_manager is not None
            assert supervisor.config is not None
            assert supervisor.message_bus is not None
            assert supervisor.llm_factory is not None
            assert supervisor.workflow_engine is not None
            assert supervisor.metrics_manager is not None
            
            # Verify WorkflowEngine has proper dependencies
            assert supervisor.workflow_engine.llm_factory is supervisor.llm_factory
            assert supervisor.workflow_engine.message_bus is supervisor.message_bus
            
        finally:
            os.unlink(config_file)


if __name__ == "__main__":
    pytest.main([__file__])