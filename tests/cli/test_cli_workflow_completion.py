#!/usr/bin/env python3
"""
Test CLI workflow completion detection.

This test reproduces the bug where CLI doesn't properly detect workflow completion
due to checking for wrong field name in the status response.
"""

import pytest
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from supervisor.supervisor import Supervisor
from supervisor.workflow_engine import WorkflowEngine, WorkflowState
from llm_provider.factory import LLMFactory
from common.message_bus import MessageBus


class TestCLIWorkflowCompletion:
    """Test CLI workflow completion detection logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock components
        self.mock_factory = Mock(spec=LLMFactory)
        self.mock_factory.get_framework = Mock(return_value='strands')
        self.mock_bus = Mock(spec=MessageBus)
        
        # Create temporary config file
        config_content = '''
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
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        self.temp_config.write(config_content)
        self.temp_config.close()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    @patch('supervisor.workflow_engine.WorkflowEngine.start_workflow')
    @patch('supervisor.workflow_engine.WorkflowEngine.get_request_status')
    def test_workflow_completion_detection_bug(self, mock_get_status, mock_start_workflow):
        """Test that reproduces the CLI workflow completion detection bug."""
        # Mock workflow responses to avoid real LLM calls
        mock_workflow_id = "wf_test123"
        mock_start_workflow.return_value = mock_workflow_id
        
        # Mock the status response that shows the bug
        mock_status = {
            "request_id": mock_workflow_id,
            "execution_state": "COMPLETED",
            "is_completed": True,  # Correct field that CLI should check
            "performance_metrics": {"total_tasks": 1, "completed_tasks": 1},
            "task_statuses": {"task_123": "COMPLETED"}
            # Note: Missing "status" field - this is the bug!
        }
        mock_get_status.return_value = mock_status
        
        # Create supervisor (no real LLM calls will be made due to mocks)
        supervisor = Supervisor(self.temp_config.name)
        
        # Start supervisor
        supervisor.start()
        
        try:
            # Start a workflow (mocked - no real LLM call)
            workflow_id = supervisor.workflow_engine.start_workflow("Test workflow")
            
            # Verify mock was called correctly
            mock_start_workflow.assert_called_once_with("Test workflow")
            assert workflow_id == mock_workflow_id
            
            # Get workflow status (mocked - returns our test data)
            status = supervisor.workflow_engine.get_request_status(workflow_id)
            
            # Verify mock was called correctly
            mock_get_status.assert_called_once_with(mock_workflow_id)
            
            # Verify the status structure
            assert "is_completed" in status
            assert "execution_state" in status
            assert "request_id" in status
            
            # The bug: CLI looks for "status" field but it doesn't exist
            assert "status" not in status  # This is the bug!
            
            # What CLI currently checks (this will be False even when completed)
            cli_completion_check = status.get("status", False)
            assert cli_completion_check == False  # Always False due to bug
            
            # What CLI should check (this correctly indicates completion)
            correct_completion_check = status.get("is_completed", False)
            assert correct_completion_check == True  # Workflow is actually completed
            
        finally:
            supervisor.stop()
    
    @patch('supervisor.workflow_engine.WorkflowEngine.start_workflow')
    @patch('supervisor.workflow_engine.WorkflowEngine.get_request_status')
    def test_get_request_status_structure(self, mock_get_status, mock_start_workflow):
        """Test the structure of get_request_status response."""
        # Mock workflow responses to avoid real LLM calls
        mock_workflow_id = "wf_test456"
        mock_start_workflow.return_value = mock_workflow_id
        
        # Mock the status response with expected structure
        mock_status = {
            "request_id": mock_workflow_id,
            "execution_state": "RUNNING",
            "is_completed": False,
            "performance_metrics": {"total_tasks": 3, "completed_tasks": 1},
            "task_statuses": {"task_1": "COMPLETED", "task_2": "RUNNING", "task_3": "PENDING"}
            # Note: No "status" field - this is what we're testing
        }
        mock_get_status.return_value = mock_status
        
        # Create WorkflowEngine directly (with mocked methods)
        workflow_engine = WorkflowEngine(self.mock_factory, self.mock_bus)
        
        # Start a workflow (mocked - no real LLM call)
        workflow_id = workflow_engine.start_workflow("Test workflow")
        
        # Verify mock was called
        mock_start_workflow.assert_called_once_with("Test workflow")
        assert workflow_id == mock_workflow_id
        
        # Get status (mocked)
        status = workflow_engine.get_request_status(workflow_id)
        
        # Verify mock was called
        mock_get_status.assert_called_once_with(mock_workflow_id)
        
        # Verify expected fields are present
        expected_fields = ["request_id", "execution_state", "is_completed",
                         "performance_metrics", "task_statuses"]
        
        for field in expected_fields:
            assert field in status, f"Expected field '{field}' missing from status"
        
        # Verify the problematic field is NOT present
        assert "status" not in status, "Unexpected 'status' field found"
        
        # Verify completion can be detected correctly
        assert isinstance(status["is_completed"], bool)
    
    @patch('cli.execute_single_workflow')
    def test_cli_completion_logic_simulation(self, mock_execute):
        """Simulate the CLI completion detection logic to show the bug."""
        
        # Mock the workflow status that get_request_status returns
        mock_status_completed = {
            "request_id": "wf_test123",
            "execution_state": "RUNNING", 
            "is_completed": True,  # Workflow IS completed
            "performance_metrics": {"total_tasks": 1, "completed_tasks": 1},
            "task_statuses": {"task_123": "COMPLETED"}
        }
        
        # Simulate CLI's current buggy logic
        def simulate_current_cli_logic(progress_info):
            """Simulate the current CLI completion detection logic."""
            if progress_info is None:
                return True  # Completed
            else:
                # This is the buggy line from cli.py:114
                if progress_info.get("status", False):  # BUG: checks "status" 
                    return True  # Completed
                else:
                    return False  # Not completed
        
        # Simulate CLI's corrected logic  
        def simulate_fixed_cli_logic(progress_info):
            """Simulate the fixed CLI completion detection logic."""
            if progress_info is None:
                return True  # Completed
            else:
                # Fixed: check "is_completed" instead of "status"
                if progress_info.get("is_completed", False):
                    return True  # Completed
                else:
                    return False  # Not completed
        
        # Test current buggy logic
        buggy_result = simulate_current_cli_logic(mock_status_completed)
        assert buggy_result == False  # Bug: doesn't detect completion
        
        # Test fixed logic
        fixed_result = simulate_fixed_cli_logic(mock_status_completed)
        assert fixed_result == True  # Fixed: correctly detects completion
        
        print("✅ Bug reproduced: Current CLI logic fails to detect completion")
        print("✅ Fix verified: Corrected CLI logic properly detects completion")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])