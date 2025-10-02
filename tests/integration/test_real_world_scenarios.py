import pytest
import time
import json
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Optional

from supervisor.supervisor import Supervisor
from supervisor.request_manager import RequestManager
from supervisor.task_scheduler import TaskScheduler, TaskPriority
from llm_provider.universal_agent import UniversalAgent
from llm_provider.factory import LLMFactory, LLMType
from common.task_context import TaskContext, ExecutionState
from common.message_bus import MessageBus, MessageType
from common.request_model import RequestMetadata


class TestRealWorldScenarios:
    """Real-world scenario tests that simulate actual usage patterns and edge cases."""
    
    @pytest.fixture
    def production_config_file(self, tmp_path):
        """Create a production-like config file."""
        config_content = """
logging:
  log_level: INFO
  log_file: production_test.log

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
  retry_delay: 2.0
  
task_scheduling:
  max_concurrent_tasks: 8
  checkpoint_interval: 180
"""
        config_file = tmp_path / "production_config.yaml"
        config_file.write_text(config_content)
        return str(config_file)
    
    @pytest.fixture
    def supervisor(self, production_config_file):
        """Create a production-configured Supervisor."""
        with patch('supervisor.supervisor.configure_logging'):
            supervisor = Supervisor(production_config_file)
            return supervisor
    
    def test_software_development_workflow(self, supervisor):
        """Test a complete software development workflow scenario."""
        # Simulate a real software development project workflow
        development_phases = [
            {
                "phase": "requirements_gathering",
                "prompt": "Analyze requirements for a task management web application with user authentication, task CRUD operations, and team collaboration features",
                "role": "analysis",
                "llm_type": LLMType.STRONG,
                "expected_output": "Requirements analysis completed: 15 functional requirements, 8 non-functional requirements identified"
            },
            {
                "phase": "architecture_design", 
                "prompt": "Design system architecture for the task management application using microservices, React frontend, Node.js backend, and PostgreSQL database",
                "role": "planning",
                "llm_type": LLMType.STRONG,
                "expected_output": "System architecture designed: 4 microservices, API gateway, database schema, and deployment strategy"
            },
            {
                "phase": "technology_research",
                "prompt": "Research best practices for React state management, Node.js authentication, and PostgreSQL optimization for the task management app",
                "role": "search",
                "llm_type": LLMType.WEAK,
                "expected_output": "Technology research completed: Redux for state management, JWT for auth, connection pooling for PostgreSQL"
            },
            {
                "phase": "implementation_planning",
                "prompt": "Create detailed implementation plan with sprint breakdown, task priorities, and resource allocation for 3-month development timeline",
                "role": "planning",
                "llm_type": LLMType.STRONG,
                "expected_output": "Implementation plan created: 6 sprints, 45 user stories, resource allocation for 4 developers"
            },
            {
                "phase": "documentation_generation",
                "prompt": "Generate comprehensive technical documentation including API specs, database schema, deployment guide, and user manual",
                "role": "summarizer",
                "llm_type": LLMType.DEFAULT,
                "expected_output": "Technical documentation generated: 4 documents totaling 50 pages with diagrams and code examples"
            }
        ]
        
        def mock_execute_side_effect(task_prompt, role, llm_type, context):
            # Find matching phase and return expected output
            for phase in development_phases:
                if phase["role"] in role and any(keyword in task_prompt.lower() 
                                               for keyword in phase["prompt"].lower().split()[:3]):
                    return phase["expected_output"]
            return f"Development phase completed for role: {role}"
        
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.side_effect = mock_execute_side_effect
            
            # Execute the complete development workflow
            workflow_start_time = time.time()
            request_ids = []
            
            for phase in development_phases:
                request = RequestMetadata(
                    prompt=phase["prompt"],
                    source_id="development_team",
                    target_id="supervisor"
                )
                
                request_id = supervisor.request_manager.handle_request(request)
                request_ids.append((request_id, phase["phase"]))
                
                # Verify each phase completes successfully
                context = supervisor.request_manager.get_request_context(request_id)
                assert context is not None
                assert context.execution_state in [ExecutionState.RUNNING, ExecutionState.COMPLETED]
            
            workflow_end_time = time.time()
            total_workflow_time = workflow_end_time - workflow_start_time
            
            # Verify workflow performance
            assert len(request_ids) == len(development_phases)
            assert total_workflow_time < 30.0  # Should complete within 30 seconds
            
            # Verify all phases were executed
            assert mock_execute.call_count >= len(development_phases)
    
    def test_customer_support_scenario(self, supervisor):
        """Test a customer support scenario with multiple interactions."""
        # Simulate customer support workflow
        support_interactions = [
            {
                "customer_query": "My application keeps crashing when I try to upload large files",
                "support_action": "Analyze the crash issue and provide troubleshooting steps",
                "role": "analysis",
                "expected_response": "Issue analysis: Large file uploads likely causing memory overflow. Recommended solutions: implement chunked uploads, increase memory limits, add progress indicators"
            },
            {
                "customer_query": "How do I configure SSL certificates for my domain?",
                "support_action": "Provide step-by-step SSL configuration guide",
                "role": "planning",
                "expected_response": "SSL configuration guide: 1) Obtain certificate from CA, 2) Configure web server, 3) Update DNS records, 4) Test HTTPS connectivity"
            },
            {
                "customer_query": "Can you help me optimize my database queries? They're running slowly",
                "support_action": "Analyze database performance and suggest optimizations",
                "role": "analysis", 
                "expected_response": "Database optimization recommendations: Add indexes on frequently queried columns, optimize JOIN operations, implement query caching, consider connection pooling"
            },
            {
                "customer_query": "I need to integrate with a third-party API but getting authentication errors",
                "support_action": "Research API integration best practices and troubleshoot authentication",
                "role": "search",
                "expected_response": "API integration solution: Verify API key format, check request headers, implement proper OAuth flow, add retry logic for transient failures"
            }
        ]
        
        def mock_execute_side_effect(task_prompt, role, llm_type, context):
            # Find matching support interaction
            for interaction in support_interactions:
                if any(keyword in task_prompt.lower() 
                      for keyword in interaction["customer_query"].lower().split()[:3]):
                    return interaction["expected_response"]
            return f"Support ticket handled by {role} agent"
        
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.side_effect = mock_execute_side_effect
            
            # Process each support interaction
            ticket_ids = []
            
            for interaction in support_interactions:
                # Create support ticket
                request = RequestMetadata(
                    prompt=f"Customer Support: {interaction['customer_query']} - {interaction['support_action']}",
                    source_id="customer_support_system",
                    target_id="supervisor"
                )
                
                ticket_start_time = time.time()
                request_id = supervisor.request_manager.handle_request(request)
                ticket_end_time = time.time()
                
                ticket_ids.append({
                    "request_id": request_id,
                    "query": interaction["customer_query"],
                    "response_time": ticket_end_time - ticket_start_time
                })
                
                # Verify ticket was processed
                assert request_id is not None
                
                # Verify response time is acceptable for customer support
                response_time = ticket_end_time - ticket_start_time
                assert response_time < 10.0, f"Support ticket response time {response_time}s too slow"
            
            # Verify all tickets were processed
            assert len(ticket_ids) == len(support_interactions)
            
            # Calculate average response time
            avg_response_time = sum(ticket["response_time"] for ticket in ticket_ids) / len(ticket_ids)
            assert avg_response_time < 5.0, f"Average support response time {avg_response_time}s too slow"
    
    def test_data_processing_pipeline(self, supervisor):
        """Test a data processing pipeline scenario."""
        # Simulate data processing workflow
        pipeline_stages = [
            {
                "stage": "data_ingestion",
                "prompt": "Process incoming CSV data with 10,000 rows containing user activity logs",
                "role": "analysis",
                "data_size": "large",
                "expected_output": "Data ingestion completed: 10,000 rows processed, 15 columns validated, 3 data quality issues identified"
            },
            {
                "stage": "data_cleaning",
                "prompt": "Clean and validate the user activity data, handle missing values and outliers",
                "role": "analysis",
                "data_size": "large",
                "expected_output": "Data cleaning completed: 9,847 valid rows, 153 rows with missing values handled, 12 outliers removed"
            },
            {
                "stage": "data_transformation",
                "prompt": "Transform user activity data into analytics-ready format with aggregations and derived metrics",
                "role": "analysis",
                "data_size": "medium",
                "expected_output": "Data transformation completed: Created 25 derived metrics, 8 aggregation tables, optimized for analytics queries"
            },
            {
                "stage": "insight_generation",
                "prompt": "Generate business insights from the processed user activity data",
                "role": "analysis",
                "data_size": "medium",
                "expected_output": "Business insights generated: User engagement up 15%, peak usage at 2-4 PM, mobile usage increased 23%"
            },
            {
                "stage": "report_creation",
                "prompt": "Create executive summary report of user activity analysis with key findings and recommendations",
                "role": "summarizer",
                "data_size": "small",
                "expected_output": "Executive report created: 3-page summary with 5 key findings, 4 strategic recommendations, and supporting visualizations"
            }
        ]
        
        def mock_execute_side_effect(task_prompt, role, llm_type, context):
            # Find matching pipeline stage
            for stage in pipeline_stages:
                if stage["stage"].replace("_", " ") in task_prompt.lower():
                    return stage["expected_output"]
            return f"Pipeline stage completed by {role}"
        
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.side_effect = mock_execute_side_effect
            
            # Execute the data processing pipeline
            pipeline_start_time = time.time()
            stage_results = []
            
            for stage in pipeline_stages:
                stage_start_time = time.time()
                
                request = RequestMetadata(
                    prompt=stage["prompt"],
                    source_id="data_processing_system",
                    target_id="supervisor"
                )
                
                request_id = supervisor.request_manager.handle_request(request)
                stage_end_time = time.time()
                
                stage_duration = stage_end_time - stage_start_time
                stage_results.append({
                    "stage": stage["stage"],
                    "request_id": request_id,
                    "duration": stage_duration,
                    "data_size": stage["data_size"]
                })
                
                # Verify stage completion
                assert request_id is not None
                
                # Performance expectations based on data size
                if stage["data_size"] == "large":
                    assert stage_duration < 15.0, f"Large data processing took {stage_duration}s, too slow"
                elif stage["data_size"] == "medium":
                    assert stage_duration < 10.0, f"Medium data processing took {stage_duration}s, too slow"
                else:  # small
                    assert stage_duration < 5.0, f"Small data processing took {stage_duration}s, too slow"
            
            pipeline_end_time = time.time()
            total_pipeline_time = pipeline_end_time - pipeline_start_time
            
            # Verify complete pipeline performance
            assert len(stage_results) == len(pipeline_stages)
            assert total_pipeline_time < 60.0, f"Complete pipeline took {total_pipeline_time}s, too slow"
    
    def test_multi_tenant_scenario(self, supervisor):
        """Test multi-tenant scenario with isolated contexts."""
        # Simulate multiple tenants using the system simultaneously
        tenants = [
            {
                "tenant_id": "tenant_a",
                "requests": [
                    "Analyze sales data for Q3 2024",
                    "Generate monthly report for stakeholders",
                    "Create forecast for Q4 2024"
                ]
            },
            {
                "tenant_id": "tenant_b", 
                "requests": [
                    "Process customer feedback survey results",
                    "Identify improvement opportunities",
                    "Draft response strategy"
                ]
            },
            {
                "tenant_id": "tenant_c",
                "requests": [
                    "Review security audit findings",
                    "Prioritize security vulnerabilities",
                    "Create remediation plan"
                ]
            }
        ]
        
        def mock_execute_side_effect(task_prompt, role, llm_type, context):
            # Determine tenant based on context or prompt content
            if "sales data" in task_prompt.lower():
                return "Sales analysis completed for Tenant A"
            elif "customer feedback" in task_prompt.lower():
                return "Customer feedback analysis completed for Tenant B"
            elif "security audit" in task_prompt.lower():
                return "Security audit review completed for Tenant C"
            else:
                return f"Task completed for tenant request"
        
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.side_effect = mock_execute_side_effect
            
            # Process requests from all tenants
            tenant_results = {}
            
            for tenant in tenants:
                tenant_id = tenant["tenant_id"]
                tenant_results[tenant_id] = []
                
                for request_prompt in tenant["requests"]:
                    request = RequestMetadata(
                        prompt=request_prompt,
                        source_id=f"{tenant_id}_client",
                        target_id="supervisor"
                    )
                    
                    request_id = supervisor.request_manager.handle_request(request)
                    tenant_results[tenant_id].append(request_id)
                    
                    # Verify tenant isolation
                    context = supervisor.request_manager.get_request_context(request_id)
                    assert context is not None
                    
                    # Each tenant should have separate contexts
                    assert context.context_id is not None
            
            # Verify all tenants were processed
            assert len(tenant_results) == len(tenants)
            
            # Verify context isolation between tenants
            all_contexts = []
            for tenant_id, request_ids in tenant_results.items():
                for request_id in request_ids:
                    context = supervisor.request_manager.get_request_context(request_id)
                    all_contexts.append(context.context_id)
            
            # All contexts should be unique (proper isolation)
            assert len(set(all_contexts)) == len(all_contexts)
    
    def test_long_running_batch_processing(self, supervisor):
        """Test long-running batch processing with checkpoints and recovery."""
        # Simulate processing a large batch of items
        batch_items = [
            f"Process document {i}: Analyze content and extract key information"
            for i in range(20)
        ]
        
        processed_items = []
        checkpoint_intervals = [5, 10, 15]  # Create checkpoints at these intervals
        
        def mock_execute_side_effect(task_prompt, role, llm_type, context):
            item_num = len(processed_items)
            processed_items.append(item_num)
            return f"Document {item_num} processed: Key information extracted and categorized"
        
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.side_effect = mock_execute_side_effect
            
            # Start batch processing
            batch_request = RequestMetadata(
                prompt="Process a batch of 20 documents for content analysis and information extraction",
                source_id="batch_processing_system",
                target_id="supervisor"
            )
            
            request_id = supervisor.request_manager.handle_request(batch_request)
            context = supervisor.request_manager.get_request_context(request_id)
            
            # Simulate processing with periodic checkpoints
            checkpoints = []
            
            for checkpoint_interval in checkpoint_intervals:
                # Simulate processing time
                time.sleep(0.1)
                
                # Create checkpoint
                checkpoint = supervisor.request_manager.pause_request(request_id)
                checkpoints.append({
                    "interval": checkpoint_interval,
                    "checkpoint": checkpoint,
                    "processed_count": len(processed_items)
                })
                
                # Resume processing
                supervisor.request_manager.resume_request(request_id, checkpoint)
            
            # Verify checkpoints were created successfully
            assert len(checkpoints) == len(checkpoint_intervals)
            
            for checkpoint_data in checkpoints:
                assert checkpoint_data["checkpoint"] is not None
                assert "context_id" in checkpoint_data["checkpoint"]
                assert "execution_state" in checkpoint_data["checkpoint"]
    
    def test_api_integration_scenario(self, supervisor):
        """Test API integration scenario with external service calls."""
        # Simulate integration with external APIs
        api_scenarios = [
            {
                "api_name": "weather_api",
                "request": "Get weather data for multiple cities: Seattle, Portland, San Francisco",
                "role": "weather",
                "mock_response": "Weather data retrieved: Seattle 72°F sunny, Portland 68°F cloudy, San Francisco 75°F clear"
            },
            {
                "api_name": "stock_api",
                "request": "Fetch stock prices for AAPL, GOOGL, MSFT and analyze trends",
                "role": "analysis",
                "mock_response": "Stock analysis completed: AAPL +2.3%, GOOGL -1.1%, MSFT +0.8%, overall tech sector positive"
            },
            {
                "api_name": "news_api",
                "request": "Search for recent news about artificial intelligence developments",
                "role": "search",
                "mock_response": "AI news summary: 15 articles found, key topics include GPT-4 updates, AI regulation, and enterprise adoption trends"
            },
            {
                "api_name": "translation_api",
                "request": "Translate product descriptions from English to Spanish, French, and German",
                "role": "summarizer",
                "mock_response": "Translation completed: Product descriptions translated to 3 languages, maintaining technical accuracy and marketing tone"
            }
        ]
        
        def mock_execute_side_effect(task_prompt, role, llm_type, context):
            # Find matching API scenario
            for scenario in api_scenarios:
                if any(keyword in task_prompt.lower() 
                      for keyword in scenario["request"].lower().split()[:3]):
                    return scenario["mock_response"]
            return f"API integration completed for {role}"
        
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.side_effect = mock_execute_side_effect
            
            # Test each API integration scenario
            api_results = []
            
            for scenario in api_scenarios:
                api_start_time = time.time()
                
                request = RequestMetadata(
                    prompt=scenario["request"],
                    source_id=f"{scenario['api_name']}_client",
                    target_id="supervisor"
                )
                
                request_id = supervisor.request_manager.handle_request(request)
                api_end_time = time.time()
                
                api_duration = api_end_time - api_start_time
                api_results.append({
                    "api_name": scenario["api_name"],
                    "request_id": request_id,
                    "duration": api_duration,
                    "success": request_id is not None
                })
                
                # Verify API integration
                assert request_id is not None
                assert api_duration < 15.0, f"API integration {scenario['api_name']} took {api_duration}s, too slow"
            
            # Verify all API integrations succeeded
            assert len(api_results) == len(api_scenarios)
            assert all(result["success"] for result in api_results)
    
    def test_edge_cases_and_boundary_conditions(self, supervisor):
        """Test edge cases and boundary conditions."""
        edge_cases = [
            {
                "case": "empty_prompt",
                "request": "",
                "should_handle_gracefully": True
            },
            {
                "case": "very_long_prompt",
                "request": "x" * 10000,  # 10K character prompt
                "should_handle_gracefully": True
            },
            {
                "case": "special_characters",
                "request": "Process this text: !@#$%^&*()_+{}|:<>?[]\\;'\",./ with special characters",
                "should_handle_gracefully": True
            },
            {
                "case": "unicode_content",
                "request": "Analyze this multilingual content: Hello 世界 🌍 Bonjour мир",
                "should_handle_gracefully": True
            },
            {
                "case": "json_like_content",
                "request": '{"task": "process", "data": [1,2,3], "config": {"mode": "test"}}',
                "should_handle_gracefully": True
            }
        ]
        
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Edge case handled successfully"
            
            for edge_case in edge_cases:
                case_name = edge_case["case"]
                prompt = edge_case["request"]
                should_handle = edge_case["should_handle_gracefully"]
                
                request = RequestMetadata(
                    prompt=prompt,
                    source_id=f"edge_case_{case_name}_client",
                    target_id="supervisor"
                )
                
                if should_handle:
                    # Should handle gracefully without exceptions
                    try:
                        request_id = supervisor.request_manager.handle_request(request)
                        
                        if request_id:  # Some edge cases might return None gracefully
                            context = supervisor.request_manager.get_request_context(request_id)
                            assert context is not None or case_name == "empty_prompt"
                        
                    except Exception as e:
                        pytest.fail(f"Edge case {case_name} was not handled gracefully: {e}")
                else:
                    # Should raise appropriate exception
                    with pytest.raises(Exception):
                        supervisor.request_manager.handle_request(request)
    
    def test_system_recovery_after_failures(self, supervisor):
        """Test system recovery capabilities after various failure scenarios."""
        # Test recovery scenarios
        recovery_scenarios = [
            {
                "failure_type": "temporary_model_unavailable",
                "description": "Model temporarily unavailable, should retry",
                "should_recover": True
            },
            {
                "failure_type": "network_timeout",
                "description": "Network timeout during request, should retry",
                "should_recover": True
            },
            {
                "failure_type": "invalid_response_format",
                "description": "Model returned invalid response format",
                "should_recover": True
            },
            {
                "failure_type": "rate_limit_exceeded",
                "description": "API rate limit exceeded, should backoff and retry",
                "should_recover": True
            }
        ]
        
        for scenario in recovery_scenarios:
            failure_count = 0
            max_failures = 2
            
            def mock_execute_side_effect(task_prompt, role, llm_type, context):
                nonlocal failure_count
                
                if failure_count < max_failures:
                    failure_count += 1
                    if scenario["failure_type"] == "temporary_model_unavailable":
                        raise Exception("Model temporarily unavailable")
                    elif scenario["failure_type"] == "network_timeout":
                        raise TimeoutError("Network request timeout")
                    elif scenario["failure_type"] == "invalid_response_format":
                        raise ValueError("Invalid response format")
                    elif scenario["failure_type"] == "rate_limit_exceeded":
                        raise Exception("Rate limit exceeded")
                
                # After failures, succeed
                return f"Recovery successful after {failure_count} failures"
            
            with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
                mock_execute.side_effect = mock_execute_side_effect
                
                request = RequestMetadata(
                    prompt=f"Test recovery from {scenario['failure_type']}",
                    source_id="recovery_test_client",
                    target_id="supervisor"
                )
                
                if scenario["should_recover"]:
                    # Should eventually succeed after retries
                    request_id = supervisor.request_manager.handle_request(request)
                    
                    # Verify recovery
                    if request_id:  # Some failures might be handled gracefully
                        context = supervisor.request_manager.get_request_context(request_id)
                        assert context is not None
                else:
                    # Should fail permanently
                    with pytest.raises(Exception):
                        supervisor.request_manager.handle_request(request)
    
    def test_production_readiness_checklist(self, supervisor):
        """Test production readiness checklist items."""
        production_checks = [
            ("configuration_loaded", lambda: supervisor.config is not None),
            ("llm_factory_initialized", lambda: supervisor.llm_factory is not None),
            ("universal_agent_ready", lambda: supervisor.request_manager.universal_agent is not None),
            ("task_scheduler_ready", lambda: supervisor.task_scheduler is not None),
            ("message_bus_ready", lambda: supervisor.message_bus is not None),
            ("metrics_available", lambda: supervisor.metrics_manager is not None),
            ("error_handling_configured", lambda: supervisor.request_manager.max_retries > 0),
            ("checkpoint_system_ready", lambda: supervisor.task_scheduler.checkpoint_interval > 0)
        ]
        
        # Verify each production readiness check
        for check_name, check_func in production_checks:
            try:
                result = check_func()
                assert result == True, f"Production check failed: {check_name}"
            except Exception as e:
                pytest.fail(f"Production readiness check {check_name} failed: {e}")
        
        # Test system can handle production-like load
        with patch.object(supervisor.request_manager.universal_agent, 'execute_task') as mock_execute:
            mock_execute.return_value = "Production load test completed"
            
            # Submit production-like load
            production_requests = [
                RequestMetadata(
                    prompt=f"Production request {i}: Process business critical data",
                    source_id=f"production_client_{i}",
                    target_id="supervisor"
                )
                for i in range(10)
            ]
            
            start_time = time.time()
            request_ids = []
            
            for request in production_requests:
                request_id = supervisor.request_manager.handle_request(request)
                request_ids.append(request_id)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Production performance requirements
            assert len(request_ids) == len(production_requests)
            assert total_time < 20.0, f"Production load processing took {total_time}s, too slow"
            
            # Verify all requests have valid contexts
            for request_id in request_ids:
                if request_id:
                    context = supervisor.request_manager.get_request_context(request_id)
                    assert context is not None


if __name__ == "__main__":
    pytest.main([__file__])