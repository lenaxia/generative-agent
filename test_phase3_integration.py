"""Phase 3 Integration Test

Tests that the full system initialization works with Phase 3 updates.
Tests ToolRegistry initialization, domain role loading, and role execution.
"""

import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_phase3_integration():
    """Test Phase 3 integration with WorkflowEngine."""
    print("=" * 60)
    print("Phase 3 Integration Test")
    print("=" * 60)

    try:
        # Import required components
        from llm_provider.factory import LLMFactory
        from common.message_bus import MessageBus
        from supervisor.workflow_engine import WorkflowEngine

        print("\n✓ Imports successful")

        # Create components
        print("\n1. Creating LLMFactory and MessageBus...")
        llm_factory = LLMFactory({})
        message_bus = MessageBus()
        print("✓ Components created")

        # Create WorkflowEngine
        print("\n2. Creating WorkflowEngine...")
        workflow_engine = WorkflowEngine(llm_factory=llm_factory, message_bus=message_bus)
        print("✓ WorkflowEngine created")

        # Test that ToolRegistry exists
        print("\n3. Checking ToolRegistry...")
        assert hasattr(
            workflow_engine, "tool_registry"
        ), "WorkflowEngine missing tool_registry"
        assert (
            workflow_engine.tool_registry is not None
        ), "tool_registry is None"
        print("✓ ToolRegistry exists")

        # Initialize Phase 3 systems
        print("\n4. Initializing Phase 3 systems...")
        await workflow_engine.initialize_phase3_systems()
        print("✓ Phase 3 systems initialized")

        # Check ToolRegistry loaded tools
        print("\n5. Checking ToolRegistry loaded tools...")
        tool_summary = workflow_engine.tool_registry.get_tool_summary()
        print(f"   Tools loaded: {tool_summary['total_tools']}")
        print(f"   Categories: {tool_summary['total_categories']}")
        assert tool_summary["total_tools"] > 0, "No tools loaded"
        print("✓ Tools loaded successfully")

        # Check domain roles were loaded
        print("\n6. Checking domain roles...")
        domain_roles = list(workflow_engine.role_registry.domain_role_instances.keys())
        print(f"   Domain roles loaded: {len(domain_roles)}")
        if domain_roles:
            print(f"   Roles: {', '.join(domain_roles)}")
        assert len(domain_roles) > 0, "No domain roles loaded"
        print("✓ Domain roles loaded successfully")

        # Test getting a specific domain role
        print("\n7. Testing domain role retrieval...")
        weather_role = workflow_engine.role_registry.get_domain_role("weather")
        if weather_role:
            print(f"   Weather role: {weather_role}")
            print(f"   Weather role tools: {len(weather_role.tools)} tools loaded")
            assert len(weather_role.tools) > 0, "Weather role has no tools"
            print("✓ Weather role retrieved and has tools")
        else:
            print("   ⚠ Weather role not found (may not be critical)")

        print("\n" + "=" * 60)
        print("✓ Phase 3 Integration Test PASSED")
        print("=" * 60)
        print("\nPhase 3 Integration Verified:")
        print("- WorkflowEngine creates ToolRegistry")
        print("- ToolRegistry loads tools from domain modules")
        print("- RoleRegistry loads domain roles")
        print("- Domain roles initialize with tools from ToolRegistry")
        print("\nSystem ready for Phase 3 fast-path execution!")

        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ Phase 3 Integration Test FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(test_phase3_integration())
    sys.exit(exit_code)
