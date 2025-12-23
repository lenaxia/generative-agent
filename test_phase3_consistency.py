"""Phase 3 Consistency Check

Thoroughly validates that all Phase 3 components are consistent:
1. Tool names match between tools.py and role.py
2. Role class names follow convention
3. All roles have required methods
4. Tool registry can find all required tools
"""

import asyncio
import sys
from pathlib import Path


async def check_consistency():
    print("=" * 70)
    print("PHASE 3 CONSISTENCY CHECK")
    print("=" * 70)

    issues = []

    # Check 1: Tool names in tools.py vs REQUIRED_TOOLS in role.py
    print("\n[1] Checking tool name consistency...")

    from llm_provider.factory import LLMFactory
    from common.message_bus import MessageBus
    from supervisor.workflow_engine import WorkflowEngine

    llm_factory = LLMFactory({})
    message_bus = MessageBus()
    workflow_engine = WorkflowEngine(llm_factory=llm_factory, message_bus=message_bus)
    await workflow_engine.initialize_phase3_systems()

    roles_to_check = ["weather", "calendar", "timer", "smart_home"]

    for role_name in roles_to_check:
        print(f"\n  Checking {role_name}:")

        # Get actual tools registered
        category_tools = workflow_engine.tool_registry.get_tools_by_category(role_name)
        actual_tool_names = set()
        for tool in category_tools:
            tool_name = workflow_engine.tool_registry._extract_tool_name(tool)
            full_name = f"{role_name}.{tool_name}"
            actual_tool_names.add(full_name)
        print(f"    Tools in ToolRegistry: {sorted(actual_tool_names)}")

        # Get required tools from role
        role = workflow_engine.role_registry.get_domain_role(role_name)
        if not role:
            issues.append(f"{role_name}: Role not found in registry")
            continue

        required_tools = set(role.REQUIRED_TOOLS)
        print(f"    REQUIRED_TOOLS:        {sorted(required_tools)}")

        # Check for mismatches
        missing = required_tools - actual_tool_names
        extra = actual_tool_names - required_tools

        if missing:
            issues.append(f"{role_name}: Required tools not found in registry: {missing}")
            print(f"    ✗ MISSING: {missing}")

        if extra:
            print(f"    ⚠ EXTRA (not required): {extra}")

        # Check if role loaded correct number of tools
        loaded_tools = len(role.tools)
        expected_tools = len(required_tools)
        if loaded_tools != expected_tools:
            issues.append(
                f"{role_name}: Loaded {loaded_tools} tools but requires {expected_tools}"
            )
            print(f"    ✗ Tool count mismatch: {loaded_tools}/{expected_tools}")
        else:
            print(f"    ✓ Loaded {loaded_tools}/{expected_tools} tools correctly")

    # Check 2: Role class names
    print("\n[2] Checking role class names...")

    role_files = {
        "weather": Path("roles/weather/role.py"),
        "calendar": Path("roles/calendar/role.py"),
        "timer": Path("roles/timer/role.py"),
        "smart_home": Path("roles/smart_home/role.py"),
    }

    expected_class_names = {
        "weather": "WeatherRole",
        "calendar": "CalendarRole",
        "timer": "TimerRole",
        "smart_home": "SmartHomeRole",
    }

    for role_name, role_file in role_files.items():
        if not role_file.exists():
            issues.append(f"{role_name}: Role file not found at {role_file}")
            continue

        content = role_file.read_text()
        expected_class = expected_class_names[role_name]

        if f"class {expected_class}" not in content:
            issues.append(f"{role_name}: Expected class '{expected_class}' not found")
            print(f"  ✗ {role_name}: Class '{expected_class}' not found")
        else:
            print(f"  ✓ {role_name}: Class '{expected_class}' found")

    # Check 3: Required methods in each role
    print("\n[3] Checking required methods...")

    required_methods = ["__init__", "initialize", "execute"]

    for role_name in roles_to_check:
        role = workflow_engine.role_registry.get_domain_role(role_name)
        if not role:
            continue

        missing_methods = []
        for method in required_methods:
            if not hasattr(role, method):
                missing_methods.append(method)

        if missing_methods:
            issues.append(f"{role_name}: Missing methods: {missing_methods}")
            print(f"  ✗ {role_name}: Missing {missing_methods}")
        else:
            print(f"  ✓ {role_name}: Has all required methods")

    # Check 4: REQUIRED_TOOLS class variable
    print("\n[4] Checking REQUIRED_TOOLS class variable...")

    for role_name in roles_to_check:
        role_class = workflow_engine.role_registry.domain_role_classes.get(role_name)
        if not role_class:
            issues.append(f"{role_name}: Role class not found")
            continue

        if not hasattr(role_class, "REQUIRED_TOOLS"):
            issues.append(f"{role_name}: Missing REQUIRED_TOOLS class variable")
            print(f"  ✗ {role_name}: Missing REQUIRED_TOOLS")
        elif not isinstance(role_class.REQUIRED_TOOLS, list):
            issues.append(f"{role_name}: REQUIRED_TOOLS is not a list")
            print(f"  ✗ {role_name}: REQUIRED_TOOLS is not a list")
        elif len(role_class.REQUIRED_TOOLS) == 0:
            issues.append(f"{role_name}: REQUIRED_TOOLS is empty")
            print(f"  ✗ {role_name}: REQUIRED_TOOLS is empty")
        else:
            print(f"  ✓ {role_name}: REQUIRED_TOOLS = {role_class.REQUIRED_TOOLS}")

    # Check 5: Initialization signature
    print("\n[5] Checking __init__ signatures...")

    for role_name in roles_to_check:
        role_class = workflow_engine.role_registry.domain_role_classes.get(role_name)
        if not role_class:
            continue

        import inspect

        sig = inspect.signature(role_class.__init__)
        params = list(sig.parameters.keys())

        expected_params = ["self", "tool_registry", "llm_factory"]
        if params != expected_params:
            issues.append(
                f"{role_name}: __init__ params {params} != expected {expected_params}"
            )
            print(f"  ✗ {role_name}: __init__{sig} (expected {expected_params})")
        else:
            print(f"  ✓ {role_name}: __init__ signature correct")

    # Check 6: Intent collector usage in execute()
    print("\n[6] Checking intent collector pattern in execute()...")

    for role_name in roles_to_check:
        role_file = role_files[role_name]
        content = role_file.read_text()

        required_patterns = [
            "IntentCollector()",
            "set_current_collector",
            "clear_current_collector",
        ]

        missing_patterns = []
        for pattern in required_patterns:
            if pattern not in content:
                missing_patterns.append(pattern)

        if missing_patterns:
            issues.append(f"{role_name}: Missing intent patterns: {missing_patterns}")
            print(f"  ⚠ {role_name}: Missing {missing_patterns}")
        else:
            print(f"  ✓ {role_name}: Uses intent collector pattern")

    # Final report
    print("\n" + "=" * 70)
    if issues:
        print("✗ CONSISTENCY CHECK FAILED")
        print("=" * 70)
        print(f"\nFound {len(issues)} issues:\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        return 1
    else:
        print("✓ CONSISTENCY CHECK PASSED")
        print("=" * 70)
        print("\nAll Phase 3 components are consistent!")
        return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(check_consistency())
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n✗ Consistency check failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
