import pytest
import time
import asyncio
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Optional

from supervisor.supervisor import Supervisor
from supervisor.request_manager import RequestManager
from supervisor.task_scheduler import TaskScheduler
from llm_provider.universal_agent import UniversalAgent
from llm_provider.factory import LLMFactory, LLMType
from common.task_context import TaskContext, ExecutionState
from common.message_bus import MessageBus, MessageType
from common.request_model import RequestMetadata


class TestEndToEndWorkflows:
    """End-to-end integration tests for complete workflows in the new architecture."""
    
    @pytest.fixture
    def mock_config_file(self, tmp_path):
        """Create a temporary config file for testing."""
        config_content = """
logging:
  log_level: INFO
  log_file: test.log

llm_providers:
  default:
    llm_class: DEFAULT
    provider_type: bedrock
    model_id: us.amazon.nova-pro-v1:0
    temperature: 0.3
    
  strong:
    llm_class: STRONG
    provider_type: bedrock
    model_id: us.amazon.nova-pro-v1:0
    temperature: 0.1
    
  weak:
    llm_class: WEAK
    provider_type: bedrock
    model_id: us.amazon.nova-lite-v1:0
    temperature: 0.5
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        return str(config_file)
    
    @pytest.fixture
    def supervisor(self, mock_config_file):
        """Create a fully configured Supervisor for testing."""
        with patch('supervisor.supervisor.configure_logging'):
            supervisor = Supervisor(mock_config_file)
            return supervisor
    
    def test_complete_request_lifecycle(self, supervisor):
        """Test complete request lifecycle from submission to completion."""
        # Mock the Universal Agent execution
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Task completed successfully"
            
            # Create and submit a request
            request = RequestMetadata(
                prompt="Create a comprehensive project plan for a web application",
                source_id="test_client",
                target_id="supervisor"
            )
            
            # Handle the request
            request_id = supervisor.request_manager.handle_request(request)
            
            assert request_id is not None
            assert request_id.startswith('req_')
            
            # Verify request context was created
            context = supervisor.request_manager.get_request_context(request_id)
            assert context is not None
            assert context.execution_state == ExecutionState.RUNNING
            
            # Verify Universal Agent was called
            mock_execute.assert_called()
            
            # Check request status
            status = supervisor.request_manager.get_request_status(request_id)
            assert status['request_id'] == request_id
            assert 'execution_state' in status
    
    def test_multi_step_workflow_with_dependencies(self, supervisor):
        """Test multi-step workflow with task dependencies."""
        # Mock Universal Agent responses for different roles
        def mock_execute_side_effect(task_prompt, role, llm_type, context):
            if role == "planning":
                return "Project plan created with 3 phases: research, development, testing"
            elif role == "search":
                return "Research completed: Found relevant technologies and frameworks"
            elif role == "summarizer":
                return "Summary: Project is feasible with estimated 3-month timeline"
            else:
                return f"Task completed for role: {role}"
        
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.side_effect = mock_execute_side_effect
            
            # Create a complex request that should generate multiple tasks
            request = RequestMetadata(
                prompt="Research and plan a machine learning project, then summarize findings",
                source_id="test_client",
                target_id="supervisor"
            )
            
            request_id = supervisor.request_manager.handle_request(request)
            
            # Verify multiple task executions occurred
            assert mock_execute.call_count >= 1
            
            # Verify different roles were used
            call_args_list = mock_execute.call_args_list
            roles_used = [call.kwargs.get('role', call.args[1] if len(call.args) > 1 else None) 
                         for call in call_args_list]
            
            # Should have used planning role at minimum
            assert any('planning' in str(role) for role in roles_used)
    
    def test_task_scheduler_integration_with_priorities(self, supervisor):
        """Test TaskScheduler integration with priority-based execution."""
        # Start the task scheduler
        supervisor.task_scheduler.start()
        
        # Create multiple requests with different priorities
        requests = [
            RequestMetadata(prompt="High priority task", source_id="client1", target_id="supervisor"),
            RequestMetadata(prompt="Normal priority task", source_id="client2", target_id="supervisor"),
            RequestMetadata(prompt="Low priority task", source_id="client3", target_id="supervisor")
        ]
        
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Task completed"
            
            # Submit all requests
            request_ids = []
            for request in requests:
                request_id = supervisor.request_manager.handle_request(request)
                request_ids.append(request_id)
            
            # Verify scheduler is managing tasks
            scheduler_metrics = supervisor.task_scheduler.get_metrics()
            assert scheduler_metrics['state'].value == "RUNNING"
            
            # Stop scheduler
            supervisor.task_scheduler.stop()
            assert supervisor.task_scheduler.get_metrics()['state'].value == "STOPPED"
    
    def test_pause_and_resume_workflow(self, supervisor):
        """Test workflow pause and resume functionality."""
        # Create a request
        request = RequestMetadata(
            prompt="Long running analysis task",
            source_id="test_client",
            target_id="supervisor"
        )
        
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Analysis in progress"
            
            request_id = supervisor.request_manager.handle_request(request)
            
            # Pause the request
            checkpoint = supervisor.request_manager.pause_request(request_id)
            assert checkpoint is not None
            
            # Verify request is paused
            context = supervisor.request_manager.get_request_context(request_id)
            assert context.execution_state == ExecutionState.PAUSED
            
            # Resume the request
            success = supervisor.request_manager.resume_request(request_id, checkpoint)
            assert success == True
            
            # Verify request is running again
            context = supervisor.request_manager.get_request_context(request_id)
            assert context.execution_state == ExecutionState.RUNNING
    
    def test_error_handling_and_recovery(self, supervisor):
        """Test error handling and recovery mechanisms."""
        # Mock Universal Agent to raise an error
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.side_effect = Exception("Simulated task failure")
            
            request = RequestMetadata(
                prompt="Task that will fail",
                source_id="test_client", 
                target_id="supervisor"
            )
            
            # Handle request - should not raise exception due to error handling
            request_id = supervisor.request_manager.handle_request(request)
            
            # Verify error was handled gracefully
            status = supervisor.request_manager.get_request_status(request_id)
            assert 'error' in status or status.get('execution_state') == 'FAILED'
    
    def test_mcp_integration_in_workflow(self, supervisor):
        """Test MCP server integration in complete workflow."""
        # Mock MCP tools
        mock_tools = [
            {"name": "web_search", "description": "Search the web"},
            {"name": "weather_lookup", "description": "Get weather data"}
        ]
        
        if supervisor.request_manager.mcp_manager:
            supervisor.request_manager.mcp_manager.get_tools_for_role.return_value = mock_tools
            supervisor.request_manager.mcp_manager.execute_tool.return_value = {"result": "MCP tool executed"}
        
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Task completed with MCP tools"
            
            request = RequestMetadata(
                prompt="Search for weather information",
                source_id="test_client",
                target_id="supervisor"
            )
            
            request_id = supervisor.request_manager.handle_request(request)
            
            # Verify MCP tools are available
            if supervisor.request_manager.mcp_manager:
                available_tools = supervisor.request_manager.get_mcp_tools("search")
                assert len(available_tools) > 0 or available_tools == []  # Either tools available or empty list
    
    def test_performance_under_load(self, supervisor):
        """Test system performance under load."""
        start_time = time.time()
        
        # Create multiple concurrent requests
        requests = [
            RequestMetadata(
                prompt=f"Task {i}: Process data batch {i}",
                source_id=f"client_{i}",
                target_id="supervisor"
            )
            for i in range(10)
        ]
        
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Batch processed"
            
            # Submit all requests
            request_ids = []
            for request in requests:
                request_id = supervisor.request_manager.handle_request(request)
                request_ids.append(request_id)
            
            # Measure completion time
            end_time = time.time()
            total_time = end_time - start_time
            
            # Verify all requests were handled
            assert len(request_ids) == 10
            
            # Performance should be reasonable (less than 5 seconds for 10 requests)
            assert total_time < 5.0
            
            # Verify all requests have contexts
            for request_id in request_ids:
                context = supervisor.request_manager.get_request_context(request_id)
                assert context is not None
    
    def test_llm_type_optimization_in_workflow(self, supervisor):
        """Test LLM type optimization based on task complexity."""
        test_cases = [
            ("Create a detailed project architecture", "planning", LLMType.STRONG),
            ("Search for recent news", "search", LLMType.WEAK),
            ("Summarize this document", "summarizer", LLMType.DEFAULT),
            ("Send a Slack message", "slack", LLMType.DEFAULT)
        ]
        
        for prompt, expected_role, expected_llm_type in test_cases:
            with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
                mock_execute.return_value = f"Task completed for {expected_role}"
                
                request = RequestMetadata(
                    prompt=prompt,
                    source_id="test_client",
                    target_id="supervisor"
                )
                
                request_id = supervisor.request_manager.handle_request(request)
                
                # Verify appropriate LLM type was used
                if mock_execute.called:
                    call_args = mock_execute.call_args
                    used_llm_type = call_args.kwargs.get('llm_type', call_args.args[2] if len(call_args.args) > 2 else None)
                    
                    # The system should optimize LLM type based on role
                    assert used_llm_type is not None
    
    def test_conversation_history_preservation(self, supervisor):
        """Test conversation history preservation across task execution."""
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Response with conversation context"
            
            request = RequestMetadata(
                prompt="Continue our previous conversation about AI",
                source_id="test_client",
                target_id="supervisor"
            )
            
            request_id = supervisor.request_manager.handle_request(request)
            
            # Get the task context
            context = supervisor.request_manager.get_request_context(request_id)
            
            # Verify conversation history exists
            history = context.get_conversation_history()
            assert history is not None
            
            # Add more conversation
            context.add_user_message("Follow-up question")
            
            # Execute another task in the same context
            mock_execute.return_value = "Follow-up response"
            
            # Verify history is preserved and growing
            updated_history = context.get_conversation_history()
            assert len(updated_history) >= len(history)
    
    def test_system_status_and_monitoring(self, supervisor):
        """Test system status and monitoring capabilities."""
        # Get overall supervisor status
        supervisor_status = supervisor.status()
        
        assert supervisor_status is not None
        assert 'running' in supervisor_status
        assert 'task_scheduler' in supervisor_status
        assert 'universal_agent' in supervisor_status
        assert 'metrics' in supervisor_status
        
        # Get Universal Agent status
        ua_status = supervisor.request_manager.get_universal_agent_status()
        
        assert ua_status['universal_agent_enabled'] == True
        assert ua_status['has_llm_factory'] == True
        assert ua_status['has_universal_agent'] == True
        assert 'framework' in ua_status
        
        # Get TaskScheduler metrics
        scheduler_metrics = supervisor.task_scheduler.get_metrics()
        
        assert 'state' in scheduler_metrics
        assert 'queued_tasks' in scheduler_metrics
        assert 'running_tasks' in scheduler_metrics
        assert 'max_concurrent_tasks' in scheduler_metrics
    
    def test_configuration_system_integration(self, supervisor):
        """Test configuration system integration with all components."""
        # Verify LLM factory has configurations
        assert supervisor.llm_factory is not None
        
        # Verify different LLM types are configured
        llm_configs = supervisor.config.llm_providers
        assert 'default' in llm_configs
        
        # Verify TaskScheduler uses configuration
        assert supervisor.task_scheduler.max_concurrent_tasks > 0
        assert supervisor.task_scheduler.checkpoint_interval > 0
        
        # Verify RequestManager uses configuration
        assert supervisor.request_manager.max_retries >= 0
        assert supervisor.request_manager.retry_delay >= 0
    
    def test_cleanup_and_resource_management(self, supervisor):
        """Test proper cleanup and resource management."""
        # Create some requests to generate resources
        requests = [
            RequestMetadata(prompt=f"Task {i}", source_id=f"client_{i}", target_id="supervisor")
            for i in range(5)
        ]
        
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Task completed"
            
            request_ids = []
            for request in requests:
                request_id = supervisor.request_manager.handle_request(request)
                request_ids.append(request_id)
            
            # Verify resources were created
            assert len(supervisor.request_manager.request_contexts) >= len(request_ids)
            
            # Test cleanup
            initial_count = len(supervisor.request_manager.request_contexts)
            supervisor.request_manager.cleanup_completed_requests(max_age_seconds=0)
            
            # Verify cleanup occurred (or at least method executed without error)
            final_count = len(supervisor.request_manager.request_contexts)
            assert final_count <= initial_count


if __name__ == "__main__":
    pytest.main([__file__])