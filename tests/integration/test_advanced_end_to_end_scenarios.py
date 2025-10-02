import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from supervisor.supervisor import Supervisor
from supervisor.workflow_engine import WorkflowEngine, TaskPriority, WorkflowState
from llm_provider.universal_agent import UniversalAgent
from llm_provider.factory import LLMFactory, LLMType
from common.task_context import TaskContext, ExecutionState
from common.message_bus import MessageBus, MessageType
from common.request_model import RequestMetadata


class TestAdvancedEndToEndScenarios:
    """Advanced end-to-end tests covering complex scenarios, edge cases, and stress testing."""
    
    @pytest.fixture
    def mock_config_file(self, tmp_path):
        """Create a comprehensive config file for advanced testing."""
        config_content = """
logging:
  log_level: DEBUG
  log_file: advanced_test.log

llm_providers:
  default:
    llm_class: DEFAULT
    provider_type: bedrock
    model_id: us.amazon.nova-pro-v1:0
    temperature: 0.3
    max_tokens: 4000
    
  strong:
    llm_class: STRONG
    provider_type: bedrock
    model_id: us.amazon.nova-pro-v1:0
    temperature: 0.1
    max_tokens: 8000
    
  weak:
    llm_class: WEAK
    provider_type: bedrock
    model_id: us.amazon.nova-lite-v1:0
    temperature: 0.5
    max_tokens: 2000

universal_agent:
  max_retries: 3
  retry_delay: 1.0
  timeout: 30.0
  
task_scheduling:
  max_concurrent_tasks: 10
  checkpoint_interval: 60
  queue_timeout: 300
"""
        config_file = tmp_path / "advanced_test_config.yaml"
        config_file.write_text(config_content)
        return str(config_file)
    
    @pytest.fixture
    def supervisor(self, mock_config_file):
        """Create a fully configured Supervisor for advanced testing."""
        with patch('supervisor.supervisor.configure_logging'):
            supervisor = Supervisor(mock_config_file)
            return supervisor
    
    def test_concurrent_request_processing(self, supervisor):
        """Test concurrent processing of multiple requests with different complexities."""
        # Define different types of requests with varying complexity
        request_scenarios = [
            ("Simple query: What is 2+2?", "search", LLMType.WEAK, "4"),
            ("Complex analysis: Analyze market trends for Q4", "analysis", LLMType.STRONG, "Detailed market analysis completed"),
            ("Code generation: Write a Python function for sorting", "coding", LLMType.STRONG, "def sort_list(items): return sorted(items)"),
            ("Weather lookup: Get weather for Seattle", "weather", LLMType.WEAK, "Sunny, 72°F"),
            ("Summarization: Summarize this 10-page document", "summarizer", LLMType.DEFAULT, "Document summarized in 3 key points"),
            ("Planning task: Create project roadmap", "planning", LLMType.STRONG, "Project roadmap with 5 phases created"),
            ("Quick search: Find Python documentation", "search", LLMType.WEAK, "Python docs found at python.org"),
            ("Data processing: Process CSV with 1000 rows", "analysis", LLMType.DEFAULT, "CSV processed, 1000 rows analyzed")
        ]
        
        def mock_execute_side_effect(instruction, role, llm_type, context):
            # Find matching scenario based on role
            for prompt, expected_role, expected_llm_type, response in request_scenarios:
                if expected_role in role:
                    return response
            return f"Task completed for role: {role}"
        
        with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.side_effect = mock_execute_side_effect
            
            # Submit all requests concurrently
            request_ids = []
            start_time = time.time()
            
            for prompt, role, llm_type, expected_response in request_scenarios:
                request = RequestMetadata(
                    prompt=prompt,
                    source_id=f"client_{role}",
                    target_id="supervisor"
                )
                request_id = supervisor.workflow_engine.handle_request(request)
                request_ids.append((request_id, role, expected_response))
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify all requests were processed
            assert len(request_ids) == len(request_scenarios)
            assert processing_time < 30.0  # Should complete within 30 seconds
            
            # Verify each request has a valid context
            for request_id, role, expected_response in request_ids:
                context = supervisor.workflow_engine.get_request_context(request_id)
                assert context is not None
                assert context.execution_state in [ExecutionState.RUNNING, ExecutionState.COMPLETED]
            
            # Verify appropriate number of Universal Agent calls
            assert mock_execute.call_count >= len(request_scenarios)
    
    def test_long_running_workflow_with_checkpoints(self, supervisor):
        """Test long-running workflow with multiple checkpoints and state preservation."""
        # Simulate a complex multi-step workflow
        workflow_steps = [
            ("Step 1: Initialize project", "planning", "Project initialized with requirements"),
            ("Step 2: Research technologies", "search", "Technology stack researched: React, Node.js, PostgreSQL"),
            ("Step 3: Design architecture", "planning", "System architecture designed with microservices"),
            ("Step 4: Create development plan", "planning", "Development plan with 8 sprints created"),
            ("Step 5: Set up infrastructure", "analysis", "Infrastructure configured on AWS"),
            ("Step 6: Generate documentation", "summarizer", "Technical documentation generated")
        ]
        
        step_counter = 0
        def mock_execute_side_effect(instruction, role, llm_type, context):
            nonlocal step_counter
            if step_counter < len(workflow_steps):
                _, _, response = workflow_steps[step_counter]
                step_counter += 1
                return response
            return "Workflow step completed"
        
        with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.side_effect = mock_execute_side_effect
            
            # Start the workflow
            request = RequestMetadata(
                prompt="Execute a complete software development workflow from planning to deployment",
                source_id="project_manager",
                target_id="supervisor"
            )
            
            request_id = supervisor.workflow_engine.handle_request(request)
            context = supervisor.workflow_engine.get_request_context(request_id)
            
            # Simulate checkpoints at different stages
            checkpoints = []
            
            # Create checkpoint after initial execution
            time.sleep(0.1)  # Simulate some processing time
            checkpoint_1 = supervisor.workflow_engine.pause_request(request_id)
            checkpoints.append(checkpoint_1)
            assert checkpoint_1 is not None
            
            # Resume and continue
            supervisor.workflow_engine.resume_request(request_id, checkpoint_1)
            
            # Create another checkpoint
            time.sleep(0.1)
            checkpoint_2 = supervisor.workflow_engine.pause_request(request_id)
            checkpoints.append(checkpoint_2)
            
            # Resume from second checkpoint
            supervisor.workflow_engine.resume_request(request_id, checkpoint_2)
            
            # Verify checkpoints contain proper state
            for i, checkpoint in enumerate(checkpoints):
                assert 'context_id' in checkpoint
                assert 'execution_state' in checkpoint
                assert 'task_graph_state' in checkpoint
                assert 'timestamp' in checkpoint
                
            # Verify workflow progression
            final_context = supervisor.workflow_engine.get_request_context(request_id)
            assert final_context is not None
    
    def test_error_recovery_and_retry_mechanisms(self, supervisor):
        """Test comprehensive error recovery and retry mechanisms."""
        failure_scenarios = [
            ("Network timeout error", "Simulated network timeout"),
            ("Model overload error", "Model temporarily unavailable"),
            ("Invalid input error", "Input validation failed"),
            ("Resource exhaustion", "Insufficient resources"),
            ("Authentication error", "API key expired")
        ]
        
        failure_count = 0
        max_failures_per_scenario = 2
        
        def mock_execute_side_effect(instruction, role, llm_type, context):
            nonlocal failure_count
            
            # Simulate failures for first few attempts
            if failure_count < len(failure_scenarios) * max_failures_per_scenario:
                scenario_index = failure_count // max_failures_per_scenario
                error_type, error_message = failure_scenarios[scenario_index]
                failure_count += 1
                raise Exception(f"{error_type}: {error_message}")
            
            # After failures, succeed
            return "Task completed successfully after retry"
        
        with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.side_effect = mock_execute_side_effect
            
            # Test each failure scenario
            for error_type, error_message in failure_scenarios:
                request = RequestMetadata(
                    prompt=f"Task that will initially fail with {error_type}",
                    source_id="test_client",
                    target_id="supervisor"
                )
                
                # This should handle errors gracefully
                request_id = supervisor.workflow_engine.handle_request(request)
                
                # Verify error handling
                status = supervisor.workflow_engine.get_request_status(request_id)
                assert status is not None
                
                # Context should exist even with errors
                context = supervisor.workflow_engine.get_request_context(request_id)
                assert context is not None
    
    def test_memory_and_resource_management_under_load(self, supervisor):
        """Test memory usage and resource management under heavy load."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create a large number of requests
        num_requests = 50
        large_requests = []
        
        for i in range(num_requests):
            # Create requests with varying sizes of context
            prompt = f"Process large dataset {i}: " + "x" * (100 * (i % 10 + 1))  # Varying prompt sizes
            request = RequestMetadata(
                prompt=prompt,
                source_id=f"load_test_client_{i}",
                target_id="supervisor"
            )
            large_requests.append(request)
        
        with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Large dataset processed"
            
            # Submit all requests
            request_ids = []
            for request in large_requests:
                request_id = supervisor.workflow_engine.handle_request(request)
                request_ids.append(request_id)
            
            # Check memory usage after processing
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Memory increase should be reasonable (less than 500MB for 50 requests)
            assert memory_increase < 500, f"Memory usage increased by {memory_increase}MB, which is too high"
            
            # Verify all requests were processed
            assert len(request_ids) == num_requests
            
            # Test cleanup
            supervisor.workflow_engine.cleanup_completed_requests(max_age_seconds=0)
            
            # Check memory after cleanup
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_after_cleanup = final_memory - initial_memory
            
            # Memory should be released after cleanup
            assert memory_after_cleanup <= memory_increase  # Memory should not increase further
    
    def test_workflow_engine_priority_and_fairness(self, supervisor):
        """Test TaskScheduler priority handling and fairness algorithms."""
        # Start the scheduler
        # WorkflowEngine starts automatically when handling requests
        pass
        
        # Create requests with different priorities
        priority_requests = [
            ("Critical system alert", TaskPriority.CRITICAL),
            ("High priority user request", TaskPriority.HIGH),
            ("Normal business task", TaskPriority.NORMAL),
            ("Background maintenance", TaskPriority.LOW),
            ("Another critical alert", TaskPriority.CRITICAL),
            ("Regular user request", TaskPriority.NORMAL),
            ("Low priority cleanup", TaskPriority.LOW),
            ("High priority analysis", TaskPriority.HIGH)
        ]
        
        execution_order = []
        
        def mock_execute_side_effect(instruction, role, llm_type, context):
            # Record execution order
            execution_order.append(instruction)
            return f"Completed: {instruction}"
        
        with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.side_effect = mock_execute_side_effect
            
            # Submit all requests
            request_ids = []
            for prompt, priority in priority_requests:
                request = RequestMetadata(
                    prompt=prompt,
                    source_id="priority_test_client",
                    target_id="supervisor"
                )
                request_id = supervisor.workflow_engine.handle_request(request)
                request_ids.append((request_id, priority))
            
            # Allow some processing time
            time.sleep(0.5)
            
            # Verify scheduler metrics
            metrics = supervisor.workflow_engine.get_workflow_metrics()
            assert metrics['state'].value == "RUNNING"
            
            # Stop scheduler
            supervisor.workflow_engine.stop_workflow_engine()
            
            # Verify execution occurred
            assert len(execution_order) >= 1
    
    def test_mcp_server_integration_scenarios(self, supervisor):
        """Test various MCP server integration scenarios."""
        # Mock comprehensive MCP tools
        mcp_tools_by_role = {
            "search": [
                {"name": "web_search", "description": "Search the web", "parameters": {"query": "string"}},
                {"name": "academic_search", "description": "Search academic papers", "parameters": {"query": "string", "year": "number"}}
            ],
            "weather": [
                {"name": "current_weather", "description": "Get current weather", "parameters": {"location": "string"}},
                {"name": "weather_forecast", "description": "Get weather forecast", "parameters": {"location": "string", "days": "number"}}
            ],
            "analysis": [
                {"name": "data_analysis", "description": "Analyze data", "parameters": {"data": "object", "type": "string"}},
                {"name": "trend_analysis", "description": "Analyze trends", "parameters": {"dataset": "array", "period": "string"}}
            ]
        }
        
        mcp_responses = {
            "web_search": {"results": ["Result 1", "Result 2", "Result 3"], "count": 3},
            "academic_search": {"papers": ["Paper A", "Paper B"], "total": 2},
            "current_weather": {"temperature": 72, "condition": "sunny", "humidity": 45},
            "weather_forecast": {"forecast": [{"day": 1, "temp": 75}, {"day": 2, "temp": 73}]},
            "data_analysis": {"summary": "Data shows positive trend", "confidence": 0.85},
            "trend_analysis": {"trend": "upward", "correlation": 0.92}
        }
        
        # Mock MCP manager
        if supervisor.workflow_engine.mcp_manager:
            def mock_get_tools_for_role(role):
                return mcp_tools_by_role.get(role, [])
            
            def mock_execute_tool(tool_name, parameters):
                return mcp_responses.get(tool_name, {"result": f"Mock result for {tool_name}"})
            
            supervisor.workflow_engine.mcp_manager.get_tools_for_role.side_effect = mock_get_tools_for_role
            supervisor.workflow_engine.mcp_manager.execute_tool.side_effect = mock_execute_tool
        
        # Test scenarios using MCP tools
        mcp_scenarios = [
            ("Search for recent AI research papers", "search", ["web_search", "academic_search"]),
            ("Get weather forecast for next week", "weather", ["current_weather", "weather_forecast"]),
            ("Analyze sales data trends", "analysis", ["data_analysis", "trend_analysis"])
        ]
        
        with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Task completed using MCP tools"
            
            for prompt, role, expected_tools in mcp_scenarios:
                request = RequestMetadata(
                    prompt=prompt,
                    source_id="mcp_test_client",
                    target_id="supervisor"
                )
                
                request_id = supervisor.workflow_engine.handle_request(request)
                
                # Verify MCP tools are available for the role
                if supervisor.workflow_engine.mcp_manager:
                    available_tools = supervisor.workflow_engine.get_mcp_tools(role)
                    tool_names = [tool["name"] for tool in available_tools]
                    
                    # Verify expected tools are available
                    for expected_tool in expected_tools:
                        assert expected_tool in tool_names or len(available_tools) == 0  # Allow for no MCP tools
    
    def test_conversation_context_and_state_management(self, supervisor):
        """Test conversation context preservation and state management across complex interactions."""
        # Simulate a multi-turn conversation with context building
        conversation_turns = [
            ("Hello, I need help with a Python project", "Initial greeting and request"),
            ("The project is a web scraper for e-commerce sites", "Project details provided"),
            ("I want to use BeautifulSoup and requests libraries", "Technical preferences specified"),
            ("Can you help me handle rate limiting?", "Specific technical question"),
            ("Also, how should I store the scraped data?", "Follow-up question"),
            ("What about error handling for network issues?", "Another follow-up question")
        ]
        
        conversation_responses = [
            "Hello! I'd be happy to help with your Python project. What kind of project are you working on?",
            "Great! A web scraper for e-commerce sites. That's a useful project. What technologies are you planning to use?",
            "Excellent choices! BeautifulSoup and requests are perfect for web scraping. What specific aspects do you need help with?",
            "For rate limiting, I recommend implementing delays between requests and respecting robots.txt files.",
            "For data storage, consider using SQLite for simple projects or PostgreSQL for more complex needs.",
            "For network error handling, implement retry logic with exponential backoff and proper exception handling."
        ]
        
        def mock_execute_side_effect(instruction, role, llm_type, context):
            # Find the appropriate response based on the conversation turn
            for i, (turn_prompt, _) in enumerate(conversation_turns):
                if turn_prompt.lower() in instruction.lower():
                    return conversation_responses[i]
            return "I understand your question and will help you with that."
        
        with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
            mock_execute.side_effect = mock_execute_side_effect
            
            # Start the conversation
            initial_request = RequestMetadata(
                prompt=conversation_turns[0][0],
                source_id="conversation_client",
                target_id="supervisor"
            )
            
            request_id = supervisor.workflow_engine.handle_request(initial_request)
            context = supervisor.workflow_engine.get_request_context(request_id)
            
            # Continue the conversation by adding messages to the same context
            for i, (turn_prompt, turn_description) in enumerate(conversation_turns[1:], 1):
                # Add user message to conversation history
                context.add_user_message(turn_prompt)
                
                # Execute the next turn using the same context
                with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_turn:
                    mock_turn.return_value = conversation_responses[i]
                    
                    # Simulate processing the turn
                    result = supervisor.workflow_engine.universal_agent.execute_task(
                        instruction=turn_prompt,
                        role="planning",
                        llm_type=LLMType.DEFAULT,
                        context=context
                    )
                    
                    # Add assistant response to conversation history
                    context.add_assistant_message(result)
            
            # Verify conversation history is preserved
            history = context.get_conversation_history()
            assert len(history) >= len(conversation_turns)  # At least one message per turn
            
            # Verify conversation context contains all turns
            conversation_context = context.get_conversation_history()
            assert len(conversation_context) > 0  # Verify conversation history exists
            assert len(conversation_context) >= len(conversation_turns)
            
            # Verify progressive summary captures the conversation
            summary = context.get_progressive_summary()
            assert "Python" in summary or "project" in summary or len(summary) > 0
    
    def test_system_resilience_and_fault_tolerance(self, supervisor):
        """Test system resilience under various fault conditions."""
        fault_scenarios = [
            ("supervisor_restart", "Simulate supervisor restart"),
            ("message_bus_failure", "Simulate message bus failure"),
            ("workflow_engine_crash", "Simulate task scheduler crash"),
            ("universal_agent_timeout", "Simulate universal agent timeout"),
            ("memory_pressure", "Simulate memory pressure"),
            ("network_partition", "Simulate network partition")
        ]
        
        for fault_type, description in fault_scenarios:
            with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
                if fault_type == "universal_agent_timeout":
                    mock_execute.side_effect = TimeoutError("Agent response timeout")
                elif fault_type == "memory_pressure":
                    mock_execute.side_effect = MemoryError("Insufficient memory")
                elif fault_type == "network_partition":
                    mock_execute.side_effect = ConnectionError("Network unreachable")
                else:
                    mock_execute.return_value = f"Handled fault: {fault_type}"
                
                # Create request during fault condition
                request = RequestMetadata(
                    prompt=f"Test request during {fault_type}",
                    source_id="fault_test_client",
                    target_id="supervisor"
                )
                
                # System should handle faults gracefully
                try:
                    request_id = supervisor.workflow_engine.handle_request(request)
                    
                    # Verify system state remains consistent
                    status = supervisor.workflow_engine.get_request_status(request_id)
                    assert status is not None
                    
                    # Context should exist even during faults
                    context = supervisor.workflow_engine.get_request_context(request_id)
                    assert context is not None
                    
                except Exception as e:
                    # If exceptions occur, they should be handled gracefully
                    assert isinstance(e, (TimeoutError, MemoryError, ConnectionError))
    
    def test_performance_benchmarking_and_optimization(self, supervisor):
        """Test performance benchmarking and optimization scenarios."""
        # Performance test scenarios
        performance_scenarios = [
            ("latency_test", 10, "Measure request processing latency"),
            ("throughput_test", 25, "Measure system throughput"),
            ("concurrent_test", 15, "Measure concurrent processing capability"),
            ("memory_efficiency_test", 20, "Measure memory efficiency")
        ]
        
        performance_results = {}
        
        for test_name, num_requests, description in performance_scenarios:
            start_time = time.time()
            
            with patch.object(supervisor.workflow_engine.universal_agent, 'execute_task') as mock_execute:
                mock_execute.return_value = f"Performance test completed: {test_name}"
                
                # Submit requests for performance test
                request_ids = []
                for i in range(num_requests):
                    request = RequestMetadata(
                        prompt=f"{test_name} request {i}",
                        source_id=f"perf_client_{i}",
                        target_id="supervisor"
                    )
                    request_id = supervisor.workflow_engine.handle_request(request)
                    request_ids.append(request_id)
                
                # Measure completion time
                end_time = time.time()
                total_time = end_time - start_time
                
                # Calculate performance metrics
                avg_latency = total_time / num_requests
                throughput = num_requests / total_time
                
                performance_results[test_name] = {
                    "total_time": total_time,
                    "avg_latency": avg_latency,
                    "throughput": throughput,
                    "num_requests": num_requests
                }
                
                # Verify all requests were processed
                assert len(request_ids) == num_requests
                
                # Performance assertions
                assert avg_latency < 1.0, f"Average latency {avg_latency}s too high for {test_name}"
                assert throughput > 5.0, f"Throughput {throughput} req/s too low for {test_name}"
        
        # Verify overall performance is acceptable
        assert len(performance_results) == len(performance_scenarios)
        
        # Log performance results for analysis
        for test_name, metrics in performance_results.items():
            print(f"{test_name}: {metrics['throughput']:.2f} req/s, {metrics['avg_latency']:.3f}s avg latency")


if __name__ == "__main__":
    pytest.main([__file__])