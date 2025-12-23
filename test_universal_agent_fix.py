"""Test that UniversalAgent now uses Phase 3 domain roles"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)

async def test():
    from llm_provider.factory import LLMFactory
    from common.message_bus import MessageBus
    from supervisor.workflow_engine import WorkflowEngine

    print("Initializing system...")
    llm_factory = LLMFactory({})
    message_bus = MessageBus()
    workflow_engine = WorkflowEngine(llm_factory=llm_factory, message_bus=message_bus)
    await workflow_engine.initialize_phase3_systems()

    print("\nTesting UniversalAgent with weather role...")

    # Create UniversalAgent (like the system does)
    from llm_provider.universal_agent import UniversalAgent
    universal_agent = UniversalAgent(llm_factory, role_registry=workflow_engine.role_registry)

    # Try to get an agent for weather role
    print("\nCalling _get_agent_for_role('weather')...")
    try:
        agent = universal_agent._get_agent_for_role("weather")
        print(f"✓ Agent returned")

        # Check if domain role was used
        if hasattr(universal_agent, '_current_domain_role') and universal_agent._current_domain_role:
            print(f"✅ Using Phase 3 domain role!")
            print(f"   Domain role type: {type(universal_agent._current_domain_role)}")
            print(f"   Domain role tools: {len(universal_agent._current_domain_role.tools)}")
        else:
            print(f"✗ Still using old role pattern")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test())
