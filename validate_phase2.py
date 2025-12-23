"""Phase 2 Validation Script

Validates that all Phase 2 tool migration is complete and working.
Tests that all domain tools can be imported and loaded.
"""

import sys


def validate_tool_imports():
    """Test that all domain tool modules can be imported."""
    print("=" * 60)
    print("Phase 2 Validation: Testing Tool Imports")
    print("=" * 60)

    domains = [
        ("weather", "roles.weather", "create_weather_tools"),
        ("calendar", "roles.calendar", "create_calendar_tools"),
        ("timer", "roles.timer", "create_timer_tools"),
        ("smart_home", "roles.smart_home", "create_smart_home_tools"),
        ("memory", "roles.memory", "create_memory_tools"),
        ("search", "roles.search", "create_search_tools"),
        ("notification", "roles.notification", "create_notification_tools"),
        ("planning", "roles.planning", "create_planning_tools"),
    ]

    all_passed = True

    for domain_name, module_path, function_name in domains:
        try:
            module = __import__(module_path, fromlist=[function_name])
            create_func = getattr(module, function_name)
            print(f"✓ {domain_name}: {module_path}.{function_name} imported")
        except ImportError as e:
            print(f"✗ {domain_name}: Import failed - {e}")
            all_passed = False
        except AttributeError as e:
            print(f"✗ {domain_name}: Function not found - {e}")
            all_passed = False

    return all_passed


def validate_tool_creation():
    """Test that tools can be created (with None providers for now)."""
    print("\n" + "=" * 60)
    print("Phase 2 Validation: Testing Tool Creation")
    print("=" * 60)

    # Import all tool creation functions
    from roles.weather import create_weather_tools
    from roles.calendar import create_calendar_tools
    from roles.timer import create_timer_tools
    from roles.smart_home import create_smart_home_tools
    from roles.memory import create_memory_tools
    from roles.search import create_search_tools
    from roles.notification import create_notification_tools
    from roles.planning import create_planning_tools

    test_cases = [
        ("weather", create_weather_tools, 2),  # get_current, get_forecast
        ("calendar", create_calendar_tools, 2),  # get_schedule, add_calendar_event
        ("timer", create_timer_tools, 3),  # set_timer, cancel_timer, list_timers
        ("smart_home", create_smart_home_tools, 3),  # ha_call_service, ha_get_state, ha_list_entities
        ("memory", create_memory_tools, 2),  # search_memory, get_recent_memories
        ("search", create_search_tools, 2),  # web_search, search_news
        ("notification", create_notification_tools, 1),  # send_notification
        ("planning", create_planning_tools, 0),  # Placeholder for Phase 4
    ]

    all_passed = True

    for domain_name, create_func, expected_count in test_cases:
        try:
            # Create tools with None provider (should still return list)
            tools = create_func(None)

            if not isinstance(tools, list):
                print(f"✗ {domain_name}: create function did not return list, got {type(tools)}")
                all_passed = False
                continue

            if len(tools) != expected_count:
                print(f"✗ {domain_name}: Expected {expected_count} tools, got {len(tools)}")
                all_passed = False
                continue

            print(f"✓ {domain_name}: Created {len(tools)} tools")

        except Exception as e:
            print(f"✗ {domain_name}: Tool creation failed - {e}")
            all_passed = False

    return all_passed


def validate_domain_structure():
    """Validate domain directory structure."""
    print("\n" + "=" * 60)
    print("Phase 2 Validation: Domain Structure")
    print("=" * 60)

    from pathlib import Path

    domains = [
        "weather",
        "calendar",
        "timer",
        "smart_home",
        "memory",
        "search",
        "notification",
        "planning",
    ]

    all_passed = True
    roles_dir = Path("roles")

    for domain in domains:
        domain_dir = roles_dir / domain
        tools_file = domain_dir / "tools.py"
        init_file = domain_dir / "__init__.py"

        if not domain_dir.exists():
            print(f"✗ {domain}/: Directory missing")
            all_passed = False
            continue

        if not tools_file.exists():
            print(f"✗ {domain}/tools.py: File missing")
            all_passed = False
        else:
            print(f"✓ {domain}/tools.py exists")

        if not init_file.exists():
            print(f"✗ {domain}/__init__.py: File missing")
            all_passed = False
        else:
            print(f"✓ {domain}/__init__.py exists")

    return all_passed


def main():
    """Run all Phase 2 validation tests."""
    print("\n" + "=" * 60)
    print("PHASE 2 TOOL MIGRATION VALIDATION")
    print("=" * 60 + "\n")

    results = []

    # Test imports
    results.append(("Tool Imports", validate_tool_imports()))

    # Test tool creation
    results.append(("Tool Creation", validate_tool_creation()))

    # Test domain structure
    results.append(("Domain Structure", validate_domain_structure()))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✓ All Phase 2 validation tests PASSED!")
        print("\nPhase 2 Complete:")
        print("- 8 domain tool modules created")
        print("- 15 total tools migrated")
        print("- Tool Registry initialization configured")
        print("\nReady for Phase 3: Role Updates")
        return 0
    else:
        print("\n✗ Some Phase 2 validation tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
