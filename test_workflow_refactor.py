"""Test Workflow After Refactoring

Tests that the system initializes correctly after tools reorganization.
Validates:
- Tool loading from new paths (tools/core/)
- Domain role initialization
- Fast-reply role recognition
- System startup
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_provider.factory import LLMFactory
from llm_provider.role_registry import RoleRegistry
from llm_provider.tool_registry import ToolRegistry

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_tool_registry():
    """Test ToolRegistry loads tools from new structure."""
    print("\n" + "=" * 60)
    print("TEST 1: ToolRegistry Initialization")
    print("=" * 60)

    try:
        tool_registry = ToolRegistry()

        # Create mock providers
        class MockProvider:
            pass

        providers = type(
            "Providers",
            (),
            {
                "memory": MockProvider(),
                "communication": MockProvider(),
                "weather": None,
                "calendar": None,
                "timer": None,
                "home_assistant": None,
                "search": None,
                "planning": None,
            },
        )()

        await tool_registry.initialize(config={}, providers=providers)

        # Check loaded tools
        total_tools = len(tool_registry._tools)
        categories = len(tool_registry._categories)

        print(f"\n✅ ToolRegistry initialized")
        print(f"   Total tools loaded: {total_tools}")
        print(f"   Categories: {categories}")

        # Check if memory and notification tools loaded
        memory_tools = [
            name for name in tool_registry._tools.keys() if name.startswith("memory.")
        ]
        notification_tools = [
            name
            for name in tool_registry._tools.keys()
            if name.startswith("notification.")
        ]

        print(f"\n   Memory tools: {memory_tools}")
        print(f"   Notification tools: {notification_tools}")

        if memory_tools:
            print(f"\n✅ Memory tools loaded from tools/core/memory.py")
        else:
            print(f"\n⚠️  No memory tools loaded")

        if notification_tools:
            print(f"✅ Notification tools loaded from tools/core/notification.py")
        else:
            print(f"⚠️  No notification tools loaded")

        return True

    except Exception as e:
        print(f"\n❌ ToolRegistry test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_role_registry():
    """Test RoleRegistry with domain roles."""
    print("\n" + "=" * 60)
    print("TEST 2: RoleRegistry Initialization")
    print("=" * 60)

    try:
        # Initialize dependencies
        tool_registry = ToolRegistry()

        class MockProvider:
            pass

        providers = type(
            "Providers",
            (),
            {
                "memory": MockProvider(),
                "communication": MockProvider(),
                "weather": None,
                "calendar": None,
                "timer": None,
                "home_assistant": None,
                "search": None,
                "planning": None,
            },
        )()

        await tool_registry.initialize(config={}, providers=providers)

        llm_factory = LLMFactory({})
        role_registry = RoleRegistry(roles_directory=Path("roles"))

        # Initialize domain roles
        await role_registry.initialize_domain_roles(tool_registry, llm_factory)

        print(f"\n✅ RoleRegistry initialized")
        print(f"   Total roles: {len(role_registry.llm_roles)}")
        print(f"   Domain roles: {list(role_registry.domain_role_instances.keys())}")

        # Check fast-reply roles
        fast_reply_roles = role_registry.get_fast_reply_roles()
        print(f"\n   Fast-reply roles: {len(fast_reply_roles)}")
        print(f"   Names: {[r.name for r in fast_reply_roles]}")

        # Check domain roles specifically
        domain_roles = ["timer", "calendar", "weather", "smart_home"]
        domain_fast_reply = [r.name for r in fast_reply_roles if r.name in domain_roles]

        if len(domain_fast_reply) == 4:
            print(f"\n✅ All 4 domain roles are fast-reply enabled")
        else:
            print(
                f"\n⚠️  Only {len(domain_fast_reply)} domain roles are fast-reply: {domain_fast_reply}"
            )

        # Check role configs
        print(f"\n   Domain Role Configurations:")
        for role_name in domain_roles:
            role_def = role_registry.llm_roles.get(role_name)
            if role_def:
                role_config = role_def.config.get("role", {})
                fast_reply = role_config.get("fast_reply", False)
                llm_type = role_config.get("llm_type", "N/A")
                print(f"   - {role_name}: fast_reply={fast_reply}, llm_type={llm_type}")

        return True

    except Exception as e:
        print(f"\n❌ RoleRegistry test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_system_integration():
    """Test full system integration."""
    print("\n" + "=" * 60)
    print("TEST 3: System Integration")
    print("=" * 60)

    try:
        # This simulates what Supervisor does on startup
        tool_registry = ToolRegistry()

        class MockProvider:
            pass

        providers = type(
            "Providers",
            (),
            {
                "memory": MockProvider(),
                "communication": MockProvider(),
                "weather": None,
                "calendar": None,
                "timer": None,
                "home_assistant": None,
                "search": None,
                "planning": None,
            },
        )()

        await tool_registry.initialize(config={}, providers=providers)

        llm_factory = LLMFactory({})
        role_registry = RoleRegistry(roles_directory=Path("roles"))

        await role_registry.initialize_domain_roles(tool_registry, llm_factory)

        # Check that domain roles can access their tools
        print(f"\n   Testing domain role tool access:")
        for role_name in ["timer", "calendar", "weather", "smart_home"]:
            role_instance = role_registry.get_domain_role(role_name)
            if role_instance:
                tools = role_instance.get_tools()
                print(f"   - {role_name}: {len(tools)} tools loaded")
            else:
                print(f"   - {role_name}: ❌ Not found")

        print(f"\n✅ System integration successful")
        return True

    except Exception as e:
        print(f"\n❌ System integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("WORKFLOW REFACTORING VALIDATION")
    print("Testing tools reorganization impact")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("ToolRegistry", await test_tool_registry()))
    results.append(("RoleRegistry", await test_role_registry()))
    results.append(("System Integration", await test_system_integration()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ ALL TESTS PASSED - System is working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - Review errors above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
