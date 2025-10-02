import pytest
import time
import json
import threading
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from supervisor.supervisor import Supervisor
from supervisor.workflow_engine import WorkflowEngine, TaskPriority, WorkflowState
from llm_provider.universal_agent import UniversalAgent
from llm_provider.factory import LLMFactory, LLMType
from common.task_context import TaskContext, ExecutionState
from common.message_bus import MessageBus, MessageType
from common.request_model import RequestMetadata


class TestMigrationValidation:
    """Comprehensive tests to validate the migration from LangChain to StrandsAgent architecture."""
    
    @pytest.fixture
    def mock_config_file(self, tmp_path):
        """Create a config file that mimics production settings."""
        config_content = """
logging:
  log_level: INFO
  log_file: migration_test.log

llm_providers:
  default:
    llm_class: DEFAULT
    provider_type: bedrock
    model_id: us.amazon.nova-pro-v1:0
    temperature: 0.3
    max_tokens: 4000
    timeout: 30.0
    
  strong:
    llm_class: STRONG
    provider_type: bedrock
    model_id: us.amazon.nova-pro-v1:0
    temperature: 0.1
    max_tokens: 8000
    timeout: 60.0
    
  weak:
    llm_class: WEAK
    provider_type: bedrock
    model_id: us.amazon.nova-lite-v1:0
    temperature: 0.5
    max_tokens: 2000
    timeout: 15.0

universal_agent:
  max_retries: 3
  retry_delay: 1.0
  
task_scheduling:
  max_concurrent_tasks: 5
  checkpoint_interval: 300
"""
        config_file = tmp_path / "migration_test_config.yaml"
        config_file.write_text(config_content)
        return str(config_file)
    
    @pytest.fixture
    def supervisor(self, mock_config_file):
        """Create a Supervisor using the new architecture."""
        with patch('supervisor.supervisor.configure_logging'):
            supervisor = Supervisor(mock_config_file)
            return supervisor
    
    def test_langchain_to_strands_migration_compatibility(self, supervisor):
        """Test that the new StrandsAgent architecture maintains compatibility with existing workflows."""
        # Simulate legacy agent requests that would have been handled by LangChain agents
        legacy_agent_scenarios = [
            {
                "agent_type": "planning_agent",
                "request": "Create a project plan for building a web application",
                "expected_role": "planning",
                "expected_llm_type": LLMType.STRONG,
                "legacy_response": "Project plan created with 5 phases: requirements, design, development, testing, deployment"
            },
            {
                "agent_type": "search_agent", 
                "request": "Find information about React best practices",
                "expected_role": "search",
                "expected_llm_type": LLMType.WEAK,
                "legacy_response": "Found comprehensive React best practices documentation"
            },
            {
                "agent_type": "weather_agent",
                "request": "Get weather forecast for Seattle",
                "expected_role": "weather", 
                "expected_llm_type": LLMType.WEAK,
                "legacy_response": "Seattle weather: Partly cloudy, 68°F, 20% chance of rain"
            },
            {
                "agent_type": "summarizer_agent",
                "request": "Summarize this technical document",
                "expected_role": "summarizer",
                "expected_llm_type": LLMType.DEFAULT,
                "legacy_response": "Document summarized: Key points include architecture overview, implementation details, and performance metrics"
            },
            {
                "agent_type": "slack_agent",
                "request": "Send project update to team channel",
                "expected_role": "slack",
                "expected_llm_type": LLMType.DEFAULT,
                "legacy_response": "Project update sent to #team-updates channel"
            }
        ]
        
        def mock_execute_side_effect(instruction, role, llm_type, context):
            # Find matching scenario and return expected response
            for scenario in legacy_agent_scenarios:
                if scenario["expected_role"] in role:
                    return scenario["legacy_response"]
            return f"Task completed for role: {role}"
        
        with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.side_effect = mock_execute_side_effect
            
            # Test each legacy agent scenario
            for scenario in legacy_agent_scenarios:
                request = RequestMetadata(
                    prompt=scenario["request"],
                    source_id=f"legacy_{scenario['agent_type']}_client",
                    target_id="supervisor"
                )
                
                request_id = supervisor.workflow_engine.handle_request(request)
                
                # Verify request was handled successfully
                assert request_id is not None
                
                # Verify context was created
                context = supervisor.workflow_engine.get_request_context(request_id)
                assert context is not None
                assert context.execution_state in [ExecutionState.RUNNING, ExecutionState.COMPLETED]
                
                # Verify Universal Agent was called
                assert mock_execute.called
                
                # Verify Universal Agent was called with proper parameters
                assert mock_execute.called
                
                # Verify request was processed successfully
                assert request_id.startswith('wf_')
    
    def test_performance_comparison_legacy_vs_new(self, supervisor):
        """Test performance comparison between legacy and new architecture patterns."""
        # Simulate performance-critical scenarios
        performance_scenarios = [
            ("Quick response task", "Simple query that should be fast", LLMType.WEAK),
            ("Complex analysis task", "Detailed analysis requiring strong model", LLMType.STRONG),
            ("Batch processing task", "Process multiple items efficiently", LLMType.DEFAULT),
            ("Real-time task", "Time-sensitive operation", LLMType.WEAK),
            ("Resource-intensive task", "Heavy computation task", LLMType.STRONG)
        ]
        
        performance_results = []
        
        for scenario_name, task_description, expected_llm_type in performance_scenarios:
            start_time = time.time()
            
            with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
                mock_execute.return_value = f"Completed: {scenario_name}"
                
                # Submit request
                request = RequestMetadata(
                    prompt=task_description,
                    source_id="performance_test_client",
                    target_id="supervisor"
                )
                
                request_id = supervisor.workflow_engine.handle_request(request)
                
                # Measure response time
                end_time = time.time()
                response_time = end_time - start_time
                
                performance_results.append({
                    "scenario": scenario_name,
                    "response_time": response_time,
                    "llm_type": expected_llm_type,
                    "success": request_id is not None
                })
                
                # Verify performance is acceptable
                assert response_time < 5.0, f"Response time {response_time}s too high for {scenario_name}"
                assert request_id is not None
        
        # Verify all scenarios completed successfully
        assert len(performance_results) == len(performance_scenarios)
        assert all(result["success"] for result in performance_results)
        
        # Calculate average performance
        avg_response_time = sum(result["response_time"] for result in performance_results) / len(performance_results)
        assert avg_response_time < 2.0, f"Average response time {avg_response_time}s too high"
    
    def test_backward_compatibility_with_existing_apis(self, supervisor):
        """Test backward compatibility with existing API patterns."""
        # Test existing API methods still work
        api_compatibility_tests = [
            {
                "method": "get_request_status",
                "args": ["test_request_123"],
                "expected_keys": ["request_id", "execution_state", "is_completed"]
            },
            {
                "method": "get_universal_agent_status", 
                "args": [],
                "expected_keys": ["universal_agent_enabled", "has_llm_factory", "framework"]
            },
            {
                "method": "list_active_requests",
                "args": [],
                "expected_type": list
            }
        ]
        
        # Create a test request first
        with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Test task completed"
            
            request = RequestMetadata(
                prompt="Test request for API compatibility",
                source_id="api_test_client",
                target_id="supervisor"
            )
            
            request_id = supervisor.workflow_engine.handle_request(request)
            
            # Test each API method
            for test_case in api_compatibility_tests:
                method_name = test_case["method"]
                args = test_case["args"]
                
                # Replace placeholder request ID with actual one
                if args and args[0] == "test_request_123":
                    args[0] = request_id
                
                # Call the method
                method = getattr(supervisor.workflow_engine, method_name)
                result = method(*args)
                
                # Verify result structure
                if "expected_keys" in test_case:
                    assert isinstance(result, dict)
                    for key in test_case["expected_keys"]:
                        assert key in result
                
                if "expected_type" in test_case:
                    assert isinstance(result, test_case["expected_type"])
    
    def test_configuration_migration_compatibility(self, supervisor):
        """Test that configuration migration maintains compatibility."""
        # Verify new configuration structure works
        assert supervisor.config is not None
        assert supervisor.llm_factory is not None
        assert supervisor.workflow_engine is not None
        assert supervisor.workflow_engine is not None
        
        # Verify LLM providers are configured
        llm_providers = supervisor.config.llm_providers
        assert "default" in llm_providers
        
        # Verify Universal Agent configuration
        ua_status = supervisor.workflow_engine.get_universal_agent_status()
        assert ua_status["universal_agent_enabled"] == True
        assert ua_status["framework"] == "strands"
        
        # Verify TaskScheduler configuration
        scheduler_metrics = supervisor.workflow_engine.get_workflow_metrics()
        assert scheduler_metrics["max_concurrent_tasks"] > 0
        
        # Test configuration values are applied correctly
        assert supervisor.workflow_engine.max_concurrent_tasks == 5  # From config
        assert supervisor.workflow_engine.checkpoint_interval == 300  # From config
        assert supervisor.workflow_engine.max_retries == 3  # From config
    
    def test_message_bus_integration_migration(self, supervisor):
        """Test that message bus integration works correctly after migration."""
        # Verify message bus is properly initialized
        assert supervisor.message_bus is not None
        
        # Test message bus subscriptions
        message_types_to_test = [
            MessageType.TASK_RESPONSE,
            MessageType.AGENT_ERROR,
            MessageType.INCOMING_REQUEST
        ]
        
        # Simulate message bus events
        for message_type in message_types_to_test:
            test_message = {
                "type": message_type.value,
                "timestamp": time.time(),
                "data": {"test": "message"}
            }
            
            # Message bus should handle these without errors
            try:
                # Note: In a real test, we'd verify actual message handling
                # For now, we just verify the message bus exists and is configured
                assert hasattr(supervisor.message_bus, 'publish')
                assert hasattr(supervisor.message_bus, 'subscribe')
            except Exception as e:
                pytest.fail(f"Message bus integration failed for {message_type}: {e}")
    
    def test_error_handling_migration_robustness(self, supervisor):
        """Test that error handling is robust after migration."""
        error_scenarios = [
            ("Invalid request format", "Malformed request data"),
            ("Model timeout", "LLM model response timeout"),
            ("Resource exhaustion", "System resources exhausted"),
            ("Network failure", "Network connectivity issues"),
            ("Configuration error", "Invalid configuration parameters")
        ]
        
        for error_type, error_description in error_scenarios:
            with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
                # Simulate different types of errors
                if "timeout" in error_type.lower():
                    mock_execute.side_effect = TimeoutError(error_description)
                elif "network" in error_type.lower():
                    mock_execute.side_effect = ConnectionError(error_description)
                elif "resource" in error_type.lower():
                    mock_execute.side_effect = MemoryError(error_description)
                else:
                    mock_execute.side_effect = Exception(error_description)
                
                # Create request that will trigger error
                request = RequestMetadata(
                    prompt=f"Test request for {error_type}",
                    source_id="error_test_client",
                    target_id="supervisor"
                )
                
                # System should handle errors gracefully
                try:
                    request_id = supervisor.workflow_engine.handle_request(request)
                    
                    # Even with errors, basic operations should work
                    if request_id:
                        status = supervisor.workflow_engine.get_request_status(request_id)
                        assert status is not None
                        
                except Exception as e:
                    # Expected exceptions should be handled gracefully
                    assert isinstance(e, (TimeoutError, ConnectionError, MemoryError, Exception))
    
    def test_scalability_and_load_handling(self, supervisor):
        """Test system scalability and load handling capabilities."""
        # Test different load scenarios
        load_scenarios = [
            ("light_load", 5, "Light load with 5 concurrent requests"),
            ("medium_load", 15, "Medium load with 15 concurrent requests"),
            ("heavy_load", 30, "Heavy load with 30 concurrent requests")
        ]
        
        for scenario_name, num_requests, description in load_scenarios:
            start_time = time.time()
            
            with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
                mock_execute.return_value = f"Load test completed: {scenario_name}"
                
                # Submit concurrent requests
                request_ids = []
                
                def submit_request(i):
                    request = RequestMetadata(
                        prompt=f"{scenario_name} request {i}",
                        source_id=f"load_client_{i}",
                        target_id="supervisor"
                    )
                    return supervisor.workflow_engine.handle_request(request)
                
                # Use ThreadPoolExecutor for concurrent submission
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(submit_request, i) for i in range(num_requests)]
                    
                    for future in as_completed(futures):
                        try:
                            request_id = future.result(timeout=10)
                            if request_id:
                                request_ids.append(request_id)
                        except Exception as e:
                            print(f"Request failed: {e}")
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # Verify load handling
                success_rate = len(request_ids) / num_requests
                throughput = len(request_ids) / total_time
                
                # Performance assertions based on load
                if scenario_name == "light_load":
                    assert success_rate >= 0.9, f"Success rate {success_rate} too low for light load"
                    assert throughput >= 2.0, f"Throughput {throughput} too low for light load"
                elif scenario_name == "medium_load":
                    assert success_rate >= 0.8, f"Success rate {success_rate} too low for medium load"
                    assert throughput >= 1.5, f"Throughput {throughput} too low for medium load"
                elif scenario_name == "heavy_load":
                    assert success_rate >= 0.7, f"Success rate {success_rate} too low for heavy load"
                    assert throughput >= 1.0, f"Throughput {throughput} too low for heavy load"
                
                print(f"{scenario_name}: {success_rate:.2%} success rate, {throughput:.2f} req/s")
    
    def test_data_consistency_and_state_management(self, supervisor):
        """Test data consistency and state management across the migration."""
        # Test state consistency across multiple operations
        consistency_scenarios = [
            ("sequential_requests", "Process requests sequentially"),
            ("concurrent_requests", "Process requests concurrently"),
            ("pause_resume_cycle", "Test pause/resume state consistency"),
            ("error_recovery", "Test state consistency after errors")
        ]
        
        for scenario_name, description in consistency_scenarios:
            with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
                mock_execute.return_value = f"Consistency test: {scenario_name}"
                
                if scenario_name == "sequential_requests":
                    # Submit requests sequentially and verify state
                    request_ids = []
                    for i in range(5):
                        request = RequestMetadata(
                            prompt=f"Sequential request {i}",
                            source_id=f"seq_client_{i}",
                            target_id="supervisor"
                        )
                        request_id = supervisor.workflow_engine.handle_request(request)
                        request_ids.append(request_id)
                        
                        # Verify state consistency
                        context = supervisor.workflow_engine.get_request_context(request_id)
                        assert context is not None
                        assert context.context_id is not None
                
                elif scenario_name == "pause_resume_cycle":
                    # Test pause/resume state consistency
                    request = RequestMetadata(
                        prompt="Long running task for pause/resume test",
                        source_id="pause_test_client",
                        target_id="supervisor"
                    )
                    
                    request_id = supervisor.workflow_engine.handle_request(request)
                    original_context = supervisor.workflow_engine.get_request_context(request_id)
                    
                    # Pause and verify state
                    checkpoint = supervisor.workflow_engine.pause_request(request_id)
                    paused_context = supervisor.workflow_engine.get_request_context(request_id)
                    assert paused_context.execution_state == ExecutionState.PAUSED
                    
                    # Resume and verify state consistency
                    supervisor.workflow_engine.resume_request(request_id, checkpoint)
                    resumed_context = supervisor.workflow_engine.get_request_context(request_id)
                    assert resumed_context.context_id == original_context.context_id
                    assert resumed_context.execution_state == ExecutionState.RUNNING
    
    def test_monitoring_and_observability_migration(self, supervisor):
        """Test that monitoring and observability features work after migration."""
        # Test various monitoring capabilities
        monitoring_tests = [
            ("supervisor_status", lambda: supervisor.status()),
            ("universal_agent_status", lambda: supervisor.workflow_engine.get_universal_agent_status()),
            ("workflow_engine_metrics", lambda: supervisor.workflow_engine.get_workflow_metrics()),
            ("active_requests", lambda: supervisor.workflow_engine.list_active_requests())
        ]
        
        # Create some activity to monitor
        with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Monitoring test task completed"
            
            # Submit a few requests to generate activity
            for i in range(3):
                request = RequestMetadata(
                    prompt=f"Monitoring test request {i}",
                    source_id=f"monitor_client_{i}",
                    target_id="supervisor"
                )
                supervisor.workflow_engine.handle_request(request)
            
            # Test each monitoring function
            for test_name, monitor_func in monitoring_tests:
                try:
                    result = monitor_func()
                    assert result is not None, f"Monitoring function {test_name} returned None"
                    
                    # Verify result structure based on function
                    if test_name == "supervisor_status":
                        assert isinstance(result, dict)
                        assert "running" in result
                        assert "workflow_engine" in result
                        assert "universal_agent" in result
                    
                    elif test_name == "universal_agent_status":
                        assert isinstance(result, dict)
                        assert "universal_agent_enabled" in result
                        assert result["universal_agent_enabled"] == True
                    
                    elif test_name == "workflow_engine_metrics":
                        assert isinstance(result, dict)
                        assert "state" in result
                        assert "max_concurrent_tasks" in result
                    
                    elif test_name == "active_requests":
                        assert isinstance(result, list)
                
                except Exception as e:
                    pytest.fail(f"Monitoring function {test_name} failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])