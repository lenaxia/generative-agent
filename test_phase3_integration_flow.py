"""Phase 3 Integration Flow Check

Validates the initialization flow and critical integration points:
1. Supervisor initialization order
2. RoleRegistry discovery logic
3. ToolRegistry initialization
4. Domain role loading sequence
"""

import asyncio
import sys
from pathlib import Path


async def check_integration_flow():
    print("=" * 70)
    print("PHASE 3 INTEGRATION FLOW CHECK")
    print("=" * 70)

    issues = []

    # Check 1: ToolRegistry.initialize() can be called with providers
    print("\n[1] Testing ToolRegistry initialization...")
    from llm_provider.tool_registry import ToolRegistry

    try:
        tool_registry = ToolRegistry()

        # Create mock providers
        class MockProvider:
            pass

        class Providers:
            def __init__(self):
                self.weather = MockProvider()
                self.calendar = MockProvider()
                self.redis = MockProvider()
                self.home_assistant = MockProvider()
                self.memory = MockProvider()
                self.search = MockProvider()
                self.communication = MockProvider()
                self.planning = MockProvider()

        providers = Providers()
        await tool_registry.initialize(config={}, providers=providers)

        if tool_registry.is_loaded():
            print(f"  ✓ ToolRegistry initialized with {len(tool_registry._tools)} tools")
        else:
            issues.append("ToolRegistry not marked as loaded after initialization")

    except Exception as e:
        issues.append(f"ToolRegistry initialization failed: {e}")
        print(f"  ✗ ToolRegistry initialization failed: {e}")

    # Check 2: RoleRegistry.initialize_domain_roles() works
    print("\n[2] Testing RoleRegistry.initialize_domain_roles()...")
    from llm_provider.role_registry import RoleRegistry
    from llm_provider.factory import LLMFactory

    try:
        role_registry = RoleRegistry("roles")
        role_registry.initialize_once()  # Discover roles

        llm_factory = LLMFactory({})

        # Check domain role classes were discovered
        discovered_count = len(role_registry.domain_role_classes)
        print(f"  ✓ Discovered {discovered_count} domain role classes")

        if discovered_count == 0:
            issues.append("No domain role classes discovered")

        # Initialize domain roles
        await role_registry.initialize_domain_roles(tool_registry, llm_factory)

        initialized_count = len(role_registry.domain_role_instances)
        print(f"  ✓ Initialized {initialized_count} domain role instances")

        if initialized_count != discovered_count:
            issues.append(
                f"Mismatch: {discovered_count} discovered but {initialized_count} initialized"
            )

    except Exception as e:
        issues.append(f"RoleRegistry initialization failed: {e}")
        print(f"  ✗ RoleRegistry initialization failed: {e}")
        import traceback

        traceback.print_exc()

    # Check 3: Role discovery order
    print("\n[3] Checking role discovery priority order...")

    # Check what files exist
    roles_dir = Path("roles")
    domain_roles = []
    single_file_roles = []

    for item in roles_dir.iterdir():
        if item.is_dir() and (item / "role.py").exists():
            domain_roles.append(item.name)
        if item.is_file() and item.name.startswith("core_"):
            role_name = item.stem.replace("core_", "")
            single_file_roles.append(role_name)

    print(f"  Domain-based roles: {sorted(domain_roles)}")
    print(f"  Single-file roles:  {sorted(single_file_roles)}")

    # Check for conflicts (both patterns exist for same role)
    conflicts = set(domain_roles) & set(single_file_roles)
    if conflicts:
        print(f"  ⚠ Conflicts (both patterns exist): {conflicts}")
        print(
            f"    Domain-based roles should take priority (checked in _load_role())"
        )

        # Verify priority is correct
        for conflict_role in conflicts:
            role_def = role_registry.get_role(conflict_role)
            if role_def and role_def.config.get("type") == "domain_based":
                print(f"    ✓ {conflict_role}: Domain-based role loaded (correct)")
            else:
                issues.append(
                    f"{conflict_role}: Single-file loaded instead of domain-based (wrong priority)"
                )
                print(f"    ✗ {conflict_role}: Wrong priority!")

    # Check 4: WorkflowEngine initialization sequence
    print("\n[4] Checking WorkflowEngine initialization sequence...")

    from supervisor.workflow_engine import WorkflowEngine
    from common.message_bus import MessageBus

    try:
        message_bus = MessageBus()
        workflow_engine = WorkflowEngine(
            llm_factory=llm_factory, message_bus=message_bus
        )

        # Check that tool_registry exists after construction
        if not hasattr(workflow_engine, "tool_registry"):
            issues.append("WorkflowEngine missing tool_registry attribute")
            print("  ✗ WorkflowEngine missing tool_registry")
        else:
            print("  ✓ WorkflowEngine has tool_registry attribute")

        # Check that initialize_phase3_systems method exists
        if not hasattr(workflow_engine, "initialize_phase3_systems"):
            issues.append("WorkflowEngine missing initialize_phase3_systems method")
            print("  ✗ WorkflowEngine missing initialize_phase3_systems")
        else:
            print("  ✓ WorkflowEngine has initialize_phase3_systems method")

            # Call it
            await workflow_engine.initialize_phase3_systems()
            print("  ✓ initialize_phase3_systems executed successfully")

            # Verify results
            if not workflow_engine.tool_registry.is_loaded():
                issues.append("ToolRegistry not loaded after initialize_phase3_systems")

            domain_count = len(workflow_engine.role_registry.domain_role_instances)
            if domain_count == 0:
                issues.append("No domain roles loaded after initialize_phase3_systems")
            else:
                print(f"  ✓ {domain_count} domain roles loaded")

    except Exception as e:
        issues.append(f"WorkflowEngine initialization failed: {e}")
        print(f"  ✗ WorkflowEngine initialization failed: {e}")
        import traceback

        traceback.print_exc()

    # Check 5: Supervisor calls initialize_phase3_systems
    print("\n[5] Checking Supervisor.start_async_tasks() calls Phase 3 init...")

    supervisor_file = Path("supervisor/supervisor.py")
    if not supervisor_file.exists():
        issues.append("supervisor.py not found")
    else:
        content = supervisor_file.read_text()

        # Check for initialize_phase3_systems call
        if "initialize_phase3_systems" not in content:
            issues.append("Supervisor does not call initialize_phase3_systems")
            print("  ✗ initialize_phase3_systems not called")
        else:
            print("  ✓ Supervisor calls initialize_phase3_systems")

        # Check it's in start_async_tasks
        if (
            "start_async_tasks" in content
            and "initialize_phase3_systems" in content
        ):
            # Rough check that they're related
            start_idx = content.find("async def start_async_tasks")
            phase3_idx = content.find("initialize_phase3_systems")
            if phase3_idx > start_idx and phase3_idx < start_idx + 2000:
                print("  ✓ Called in start_async_tasks method")
            else:
                print("  ⚠ May not be in start_async_tasks method")

    # Check 6: Tool creation functions exist
    print("\n[6] Checking tool creation functions...")

    tool_modules = {
        "weather": "roles/weather/tools.py",
        "calendar": "roles/calendar/tools.py",
        "timer": "roles/timer/tools.py",
        "smart_home": "roles/smart_home/tools.py",
    }

    for domain, tool_file in tool_modules.items():
        path = Path(tool_file)
        if not path.exists():
            issues.append(f"{domain}: Tool file not found at {tool_file}")
            continue

        content = path.read_text()
        expected_func = f"def create_{domain}_tools("

        if expected_func not in content:
            issues.append(f"{domain}: Missing create_{domain}_tools() function")
            print(f"  ✗ {domain}: Missing create function")
        else:
            print(f"  ✓ {domain}: Has create_{domain}_tools()")

    # Final report
    print("\n" + "=" * 70)
    if issues:
        print("✗ INTEGRATION FLOW CHECK FAILED")
        print("=" * 70)
        print(f"\nFound {len(issues)} issues:\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        return 1
    else:
        print("✓ INTEGRATION FLOW CHECK PASSED")
        print("=" * 70)
        print("\nAll integration points verified!")
        return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(check_integration_flow())
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n✗ Integration flow check failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
