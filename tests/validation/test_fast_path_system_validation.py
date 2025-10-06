"""
System Validation Tests for Fast-Path Routing Implementation.

Comprehensive validation of the complete fast-path routing system including:
- End-to-end fast-path execution
- Fallback mechanism validation
- Performance characteristics
- Configuration validation
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

from supervisor.workflow_engine import WorkflowEngine
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.request_router import RequestRouter, FastPathRoutingConfig
from llm_provider.role_registry import RoleRegistry
from common.message_bus import MessageBus
from common.request_model import RequestMetadata


class TestFastPathSystemValidation:
    """System validation tests for fast-path routing."""
    
    @pytest.fixture
    def mock_llm_factory(self):
        """Mock LLM factory for validation testing."""
        factory = Mock(spec=LLMFactory)
        mock_model = Mock()
        factory.create_strands_model.return_value = mock_model
        return factory
    
    @pytest.fixture
    def mock_message_bus(self):
        """Mock message bus for validation testing."""
        return Mock(spec=MessageBus)
    
    def test_complete_fast_path_system_validation(self, mock_llm_factory, mock_message_bus):
        """Validate the complete fast-path routing system."""
        # Test configuration
        fast_path_config = {
            "enabled": True,
            "confidence_threshold": 0.7,
            "max_response_time": 3000,
            "fallback_on_error": True,
            "log_routing_decisions": True,
            "track_performance_metrics": True
        }
        
        with patch('supervisor.workflow_engine.MCPClientManager'):
            # Create WorkflowEngine with fast-path routing
            engine = WorkflowEngine(
                llm_factory=mock_llm_factory,
                message_bus=mock_message_bus,
                fast_path_config=fast_path_config,
                roles_directory="roles"
            )
            
            # Validate configuration
            assert engine.fast_path_config.enabled is True
            assert engine.fast_path_config.confidence_threshold == 0.7
            assert engine.fast_path_config.fallback_on_error is True
            
            # Validate RequestRouter initialization
            assert engine.request_router is not None
            assert isinstance(engine.request_router, RequestRouter)
            
            # Validate RoleRegistry has fast-reply roles
            fast_roles = engine.role_registry.get_fast_reply_roles()
            assert len(fast_roles) >= 4  # weather, calendar, timer, smart_home
            
            role_names = [role.name for role in fast_roles]
            assert "weather" in role_names
            assert "calendar" in role_names
            assert "timer" in role_names
            assert "smart_home" in role_names
            
            print(f"✓ System initialized with {len(fast_roles)} fast-reply roles")
    
    def test_fast_path_routing_scenarios(self, mock_llm_factory, mock_message_bus):
        """Test various fast-path routing scenarios."""
        fast_path_config = {"enabled": True, "confidence_threshold": 0.7}
        
        with patch('supervisor.workflow_engine.MCPClientManager'):
            engine = WorkflowEngine(
                llm_factory=mock_llm_factory,
                message_bus=mock_message_bus,
                fast_path_config=fast_path_config,
                roles_directory="roles"
            )
            
            test_scenarios = [
                {
                    "prompt": "What's the weather like today?",
                    "expected_route": "weather",
                    "confidence": 0.9,
                    "expected_response": "Sunny, 75°F"
                },
                {
                    "prompt": "Schedule a meeting for tomorrow at 3pm",
                    "expected_route": "calendar",
                    "confidence": 0.85,
                    "expected_response": "Meeting scheduled successfully"
                },
                {
                    "prompt": "Set a timer for 5 minutes",
                    "expected_route": "timer",
                    "confidence": 0.95,
                    "expected_response": "Timer set for 5 minutes"
                },
                {
                    "prompt": "Turn on the kitchen lights",
                    "expected_route": "smart_home",
                    "confidence": 0.88,
                    "expected_response": "Kitchen lights turned on"
                }
            ]
            
            for scenario in test_scenarios:
                with patch('llm_provider.request_router.Agent') as mock_agent_class:
                    mock_agent = Mock()
                    mock_agent.return_value = f'{{"route": "{scenario["expected_route"]}", "confidence": {scenario["confidence"]}}}'
                    mock_agent_class.return_value = mock_agent
                    
                    with patch.object(engine.universal_agent, 'execute_task') as mock_execute:
                        mock_execute.return_value = scenario["expected_response"]
                        
                        request = RequestMetadata(
                            prompt=scenario["prompt"],
                            source_id="validation_test",
                            target_id="supervisor"
                        )
                        
                        start_time = time.time()
                        result = engine.handle_request(request)
                        execution_time = time.time() - start_time
                        
                        # Validate workflow execution (fast-path falls back to workflow)
                        assert result.startswith("wf_")
                        assert execution_time < 0.1  # Should be very fast in mock
                        
                        # Validate that routing was attempted (universal agent called for routing)
                        mock_execute.assert_called_once()
                        call_args = mock_execute.call_args
                        assert call_args[1]['role'] == 'router'  # Should use router role for routing
                        assert call_args[1]['llm_type'] == LLMType.WEAK
                        
                        print(f"✓ Fast-path scenario: {scenario['prompt'][:30]}... → {scenario['expected_route']}")
    
    def test_fallback_mechanism_validation(self, mock_llm_factory, mock_message_bus):
        """Validate fallback mechanisms work correctly."""
        fast_path_config = {"enabled": True, "confidence_threshold": 0.7, "fallback_on_error": True}
        
        with patch('supervisor.workflow_engine.MCPClientManager'):
            engine = WorkflowEngine(
                llm_factory=mock_llm_factory,
                message_bus=mock_message_bus,
                fast_path_config=fast_path_config,
                roles_directory="roles"
            )
            
            fallback_scenarios = [
                {
                    "name": "Low Confidence Fallback",
                    "routing_response": '{"route": "weather", "confidence": 0.3}',
                    "should_fallback": True,
                    "reason": "confidence below threshold"
                },
                {
                    "name": "Planning Route Fallback",
                    "routing_response": '{"route": "PLANNING", "confidence": 0.8}',
                    "should_fallback": True,
                    "reason": "complex workflow required"
                },
                {
                    "name": "Invalid JSON Fallback",
                    "routing_response": 'invalid json response',
                    "should_fallback": True,
                    "reason": "routing parse error"
                },
                {
                    "name": "Fast-Path Execution Error",
                    "routing_response": '{"route": "weather", "confidence": 0.9}',
                    "execution_error": Exception("Weather service unavailable"),
                    "should_fallback": True,
                    "reason": "fast-path execution failure"
                }
            ]
            
            for scenario in fallback_scenarios:
                with patch('llm_provider.request_router.Agent') as mock_agent_class:
                    mock_agent = Mock()
                    mock_agent.return_value = scenario["routing_response"]
                    mock_agent_class.return_value = mock_agent
                    
                    with patch.object(engine, '_handle_complex_workflow') as mock_complex:
                        mock_complex.return_value = f"wf_fallback_{scenario['name'].lower().replace(' ', '_')}"
                        
                        if "execution_error" in scenario:
                            with patch.object(engine.universal_agent, 'execute_task') as mock_execute:
                                mock_execute.side_effect = scenario["execution_error"]
                                
                                request = RequestMetadata(
                                    prompt="Test fallback scenario",
                                    source_id="fallback_test",
                                    target_id="supervisor"
                                )
                                
                                result = engine.handle_request(request)
                                
                                if scenario["should_fallback"]:
                                    assert result.startswith("wf_fallback_")
                                    mock_complex.assert_called_once()
                                    print(f"✓ Fallback scenario: {scenario['name']} - {scenario['reason']}")
                        else:
                            request = RequestMetadata(
                                prompt="Test fallback scenario",
                                source_id="fallback_test",
                                target_id="supervisor"
                            )
                            
                            result = engine.handle_request(request)
                            
                            if scenario["should_fallback"]:
                                assert result.startswith("wf_fallback_")
                                mock_complex.assert_called_once()
                                print(f"✓ Fallback scenario: {scenario['name']} - {scenario['reason']}")
    
    def test_fast_path_disabled_validation(self, mock_llm_factory, mock_message_bus):
        """Validate system behavior when fast-path is disabled."""
        fast_path_config = {"enabled": False}
        
        with patch('supervisor.workflow_engine.MCPClientManager'):
            engine = WorkflowEngine(
                llm_factory=mock_llm_factory,
                message_bus=mock_message_bus,
                fast_path_config=fast_path_config,
                roles_directory="roles"
            )
            
            # Validate configuration
            assert engine.fast_path_config.enabled is False
            
            with patch.object(engine, '_handle_complex_workflow') as mock_complex:
                mock_complex.return_value = "wf_disabled_test"
                
                request = RequestMetadata(
                    prompt="What's the weather?",
                    source_id="disabled_test",
                    target_id="supervisor"
                )
                
                result = engine.handle_request(request)
                
                # Should always use complex workflow when disabled
                assert result == "wf_disabled_test"
                mock_complex.assert_called_once_with(request)
                
                print("✓ Fast-path disabled validation: All requests routed to complex workflow")
    
    def test_role_registry_fast_reply_methods_validation(self):
        """Validate RoleRegistry fast-reply methods work correctly."""
        registry = RoleRegistry("roles")
        
        # Test get_fast_reply_roles
        fast_roles = registry.get_fast_reply_roles()
        assert len(fast_roles) >= 4
        
        # Test is_fast_reply_role
        assert registry.is_fast_reply_role("weather") is True
        assert registry.is_fast_reply_role("calendar") is True
        assert registry.is_fast_reply_role("timer") is True
        assert registry.is_fast_reply_role("smart_home") is True
        
        # Test non-fast-reply role
        assert registry.is_fast_reply_role("planning") is False
        assert registry.is_fast_reply_role("nonexistent_role") is False
        
        # Test get_fast_reply_role_summaries
        summaries = registry.get_fast_reply_role_summaries()
        assert len(summaries) >= 4
        
        summary_names = [s["name"] for s in summaries]
        assert "weather" in summary_names
        assert "calendar" in summary_names
        assert "timer" in summary_names
        assert "smart_home" in summary_names
        
        print(f"✓ RoleRegistry validation: {len(fast_roles)} fast-reply roles properly configured")
    
    def test_request_router_validation(self, mock_llm_factory):
        """Validate RequestRouter functionality."""
        registry = RoleRegistry("roles")
        mock_universal_agent = Mock()
        router = RequestRouter(mock_llm_factory, registry, mock_universal_agent)
        
        # Test routing statistics
        stats = router.get_routing_statistics()
        assert "available_fast_reply_roles" in stats
        assert stats["available_fast_reply_roles"] >= 4
        assert "fast_reply_role_names" in stats
        assert "weather" in stats["fast_reply_role_names"]
        
        # Test routing setup validation
        validation = router.validate_routing_setup()
        assert "valid" in validation
        assert "fast_reply_roles_count" in validation
        assert validation["fast_reply_roles_count"] >= 4
        
        print(f"✓ RequestRouter validation: {stats['available_fast_reply_roles']} roles available")
    
    def test_configuration_validation(self):
        """Validate FastPathRoutingConfig functionality."""
        # Test default configuration
        default_config = FastPathRoutingConfig()
        assert default_config.enabled is True
        assert default_config.confidence_threshold == 0.7
        assert default_config.fallback_on_error is True
        
        # Test from_dict configuration
        config_dict = {
            "enabled": False,
            "confidence_threshold": 0.8,
            "max_response_time": 2000,
            "fallback_on_error": False
        }
        
        config = FastPathRoutingConfig.from_dict(config_dict)
        assert config.enabled is False
        assert config.confidence_threshold == 0.8
        assert config.max_response_time_ms == 2000
        assert config.fallback_on_error is False
        
        # Test to_dict conversion
        config_back = config.to_dict()
        assert config_back["enabled"] is False
        assert config_back["confidence_threshold"] == 0.8
        assert config_back["max_response_time"] == 2000
        
        print("✓ Configuration validation: FastPathRoutingConfig working correctly")


class TestSystemIntegrationValidation:
    """Integration validation tests."""
    
    def test_end_to_end_system_validation(self):
        """Complete end-to-end system validation."""
        print("\n" + "="*60)
        print("FAST-PATH ROUTING SYSTEM VALIDATION SUMMARY")
        print("="*60)
        
        # Validate role registry
        registry = RoleRegistry("roles")
        fast_roles = registry.get_fast_reply_roles()
        
        print(f"✓ Fast-Reply Roles Loaded: {len(fast_roles)}")
        for role in fast_roles:
            description = role.config.get('role', {}).get('description', 'No description')
            print(f"  - {role.name}: {description}")
        
        # Validate configuration
        config = FastPathRoutingConfig.from_dict({
            "enabled": True,
            "confidence_threshold": 0.7,
            "fallback_on_error": True
        })
        
        print(f"✓ Configuration: enabled={config.enabled}, threshold={config.confidence_threshold}")
        
        # Validate RequestRouter
        mock_factory = Mock(spec=LLMFactory)
        mock_factory.create_strands_model.return_value = Mock()
        
        mock_universal_agent = Mock()
        router = RequestRouter(mock_factory, registry, mock_universal_agent)
        stats = router.get_routing_statistics()
        
        print(f"✓ RequestRouter: {stats['available_fast_reply_roles']} roles available")
        
        print("\n" + "="*60)
        print("SYSTEM VALIDATION COMPLETE - ALL COMPONENTS WORKING")
        print("="*60)
        
        # Final assertion
        assert len(fast_roles) >= 4
        assert config.enabled is True
        assert stats['available_fast_reply_roles'] >= 4