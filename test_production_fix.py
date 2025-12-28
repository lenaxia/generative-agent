"""Quick test to show Phase 3 production fix works"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


async def test():
    print("=" * 70)
    print("PHASE 3 PRODUCTION FIX VERIFICATION")
    print("=" * 70)

    from common.message_bus import MessageBus
    from llm_provider.factory import LLMFactory
    from llm_provider.universal_agent import UniversalAgent
    from supervisor.workflow_engine import WorkflowEngine

    # Initialize system
    print("\n[1] Initializing system with Phase 3...")
    llm_factory = LLMFactory({})
    message_bus = MessageBus()
    workflow_engine = WorkflowEngine(llm_factory=llm_factory, message_bus=message_bus)
    await workflow_engine.initialize_phase3_systems()
    print("✓ System initialized")

    # Create UniversalAgent
    print("\n[2] Creating UniversalAgent...")
    universal_agent = UniversalAgent(
        llm_factory, role_registry=workflow_engine.role_registry
    )
    print("✓ UniversalAgent created")

    # Test assume_role for weather
    print("\n[3] Testing assume_role('weather')...")
    print("    (This is what the system calls when routing to weather role)")

    try:
        agent = universal_agent.assume_role("weather")

        # Check if domain role was used
        if (
            hasattr(universal_agent, "_current_domain_role")
            and universal_agent._current_domain_role
        ):
            print("\n✅ SUCCESS: Phase 3 domain role is being used!")
            print(
                f"   Domain role type: {type(universal_agent._current_domain_role).__name__}"
            )
            print(
                f"   Tools available: {len(universal_agent._current_domain_role.tools)}"
            )
            print(
                f"   Required tools: {universal_agent._current_domain_role.REQUIRED_TOOLS}"
            )
            return 0
        else:
            print("\n✗ FAIL: Still using old role pattern")
            return 1

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(test())
    print("\n" + "=" * 70)
    if exit_code == 0:
        print("✅ Phase 3 fix verified - domain roles will be used in production!")
        print("\nNow when you run 'python3 cli.py', the weather role will use:")
        print("  - get_current_weather tool")
        print("  - get_forecast tool")
        print("\nInstead of the old calculator/file/shell tools.")
    else:
        print("✗ Phase 3 fix needs more work")
    print("=" * 70)
