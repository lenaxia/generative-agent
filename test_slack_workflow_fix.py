#!/usr/bin/env python3
"""
Test script to verify the Slack workflow fix.

This test verifies that incoming Slack messages are properly converted
to RequestMetadata objects and can trigger workflows.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

from common.communication_manager import CommunicationManager
from common.message_bus import MessageBus, MessageType
from common.request_model import RequestMetadata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockWorkflowEngine:
    """Mock workflow engine to capture incoming requests."""
    
    def __init__(self):
        self.received_requests = []
    
    def handle_request(self, request):
        """Handle incoming request and verify it's a RequestMetadata object."""
        logger.info(f"Received request: {request}")
        logger.info(f"Request type: {type(request)}")
        
        # Verify it's a RequestMetadata object
        if isinstance(request, RequestMetadata):
            logger.info("✅ Request is properly formatted as RequestMetadata")
            logger.info(f"  - Prompt: {request.prompt}")
            logger.info(f"  - Source ID: {request.source_id}")
            logger.info(f"  - Target ID: {request.target_id}")
            logger.info(f"  - Metadata: {request.metadata}")
            logger.info(f"  - Response requested: {request.response_requested}")
        else:
            logger.error(f"❌ Request is not RequestMetadata: {type(request)}")
        
        self.received_requests.append(request)
        return "test_workflow_id"


async def test_slack_message_to_workflow():
    """Test that Slack messages are properly converted to RequestMetadata."""
    logger.info("🧪 Testing Slack message to workflow conversion...")
    
    # Create message bus and communication manager
    message_bus = MessageBus()
    message_bus.start()
    
    # Create mock workflow engine
    mock_workflow_engine = MockWorkflowEngine()
    
    # Subscribe mock workflow engine to INCOMING_REQUEST
    message_bus.subscribe(
        mock_workflow_engine, 
        MessageType.INCOMING_REQUEST, 
        mock_workflow_engine.handle_request
    )
    
    # Create communication manager (without auto-discovery)
    comm_manager = CommunicationManager(message_bus)
    comm_manager._discover_and_initialize_channels = AsyncMock()
    await comm_manager.initialize()
    
    # Simulate a Slack message
    slack_message = {
        "type": "app_mention",
        "user_id": "U12345",
        "channel_id": "C67890", 
        "text": "Hello, can you help me with a task?",
        "timestamp": "1234567890.123"
    }
    
    logger.info(f"📨 Simulating Slack message: {slack_message}")
    
    # Process the message through the communication manager
    await comm_manager._handle_channel_message("slack", slack_message)
    
    # Give some time for async processing
    await asyncio.sleep(0.1)
    
    # Verify the results
    try:
        if mock_workflow_engine.received_requests:
            request = mock_workflow_engine.received_requests[0]
            if isinstance(request, RequestMetadata):
                logger.info("✅ SUCCESS: Slack message properly converted to RequestMetadata")
                logger.info(f"   Prompt: '{request.prompt}'")
                logger.info(f"   Source: {request.source_id}")
                logger.info(f"   Target: {request.target_id}")
                return True
            else:
                logger.error(f"❌ FAIL: Received {type(request)} instead of RequestMetadata")
                return False
        else:
            logger.error("❌ FAIL: No requests received by workflow engine")
            return False
    finally:
        message_bus.stop()


async def main():
    """Run the test."""
    logger.info("🚀 Starting Slack workflow fix test...")
    
    try:
        success = await test_slack_message_to_workflow()
        if success:
            logger.info("🎉 All tests passed! Slack messages should now trigger workflows.")
        else:
            logger.error("💥 Test failed! The fix may need additional work.")
        return success
    except Exception as e:
        logger.error(f"💥 Test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)