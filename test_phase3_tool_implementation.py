"""Phase 3 Tool Implementation Check

Validates that tool implementations are correct:
1. Tools are properly decorated with @tool
2. Tools have docstrings
3. Tools have correct function signatures
4. Tool creation functions return lists
"""

import asyncio
import inspect
import sys
from pathlib import Path


async def check_tool_implementations():
    print("=" * 70)
    print("PHASE 3 TOOL IMPLEMENTATION CHECK")
    print("=" * 70)

    issues = []

    # Initialize to get actual tools
    from common.message_bus import MessageBus
    from llm_provider.factory import LLMFactory
    from supervisor.workflow_engine import WorkflowEngine

    llm_factory = LLMFactory({})
    message_bus = MessageBus()
    workflow_engine = WorkflowEngine(llm_factory=llm_factory, message_bus=message_bus)
    await workflow_engine.initialize_phase3_systems()

    # Check each domain
    domains = ["weather", "calendar", "timer", "smart_home"]

    for domain in domains:
        print(f"\n[{domain.upper()}] Checking tools...")

        tools = workflow_engine.tool_registry.get_tools_by_category(domain)
        print(f"  Found {len(tools)} tools")

        if len(tools) == 0:
            issues.append(f"{domain}: No tools found")
            continue

        for tool in tools:
            tool_name = workflow_engine.tool_registry._extract_tool_name(tool)

            # Check 1: Tool has name
            if not tool_name:
                issues.append(f"{domain}: Tool has no name")
                print(f"  ✗ Tool has no name")
                continue

            # Check 2: Tool has description/docstring
            description = workflow_engine.tool_registry._extract_tool_description(tool)
            if not description:
                issues.append(f"{domain}.{tool_name}: Missing description/docstring")
                print(f"  ⚠ {tool_name}: No description")
            else:
                print(f"  ✓ {tool_name}: Has description")

            # Check 3: Tool has callable function
            if not callable(tool):
                # Check if it's a Strands Tool object
                if hasattr(tool, "fn") and callable(tool.fn):
                    print(f"  ✓ {tool_name}: Has callable function (Strands Tool)")
                else:
                    issues.append(f"{domain}.{tool_name}: Not callable")
                    print(f"  ✗ {tool_name}: Not callable")

    # Check tool creation functions
    print("\n[TOOL CREATION FUNCTIONS]")

    tool_modules = {
        "weather": "roles.weather.tools",
        "calendar": "roles.calendar.tools",
        "timer": "roles.timer.tools",
        "smart_home": "roles.smart_home.tools",
    }

    for domain, module_path in tool_modules.items():
        print(f"\n  {domain}:")

        try:
            import importlib

            module = importlib.import_module(module_path)

            # Find creation function
            creation_func_name = f"create_{domain}_tools"
            if not hasattr(module, creation_func_name):
                issues.append(f"{domain}: Missing {creation_func_name}()")
                print(f"    ✗ Missing {creation_func_name}()")
                continue

            creation_func = getattr(module, creation_func_name)
            print(f"    ✓ Has {creation_func_name}()")

            # Check signature
            sig = inspect.signature(creation_func)
            params = list(sig.parameters.keys())
            if len(params) != 1:
                issues.append(
                    f"{domain}: {creation_func_name} should take 1 parameter (provider), got {params}"
                )
                print(f"    ⚠ Parameters: {params} (expected 1 provider param)")
            else:
                print(f"    ✓ Correct signature: {sig}")

            # Check return type annotation if exists
            if sig.return_annotation != inspect.Signature.empty:
                ret_type = str(sig.return_annotation)
                if "list" not in ret_type.lower():
                    print(f"    ⚠ Return type: {ret_type} (should be list)")
            else:
                print(f"    ⚠ No return type annotation")

        except Exception as e:
            issues.append(f"{domain}: Failed to import/check: {e}")
            print(f"    ✗ Error: {e}")

    # Check for @tool decorator usage
    print("\n[TOOL DECORATORS]")

    for domain, module_path in tool_modules.items():
        file_path = Path(module_path.replace(".", "/") + ".py")
        if not file_path.exists():
            continue

        content = file_path.read_text()
        decorator_count = content.count("@tool")
        print(f"  {domain}: {decorator_count} @tool decorators found")

        if decorator_count == 0:
            issues.append(f"{domain}: No @tool decorators found in {file_path}")

    # Check specific tool signatures
    print("\n[SPECIFIC TOOL SIGNATURES]")

    # Weather tools should be async
    print("\n  Weather tools:")
    weather_tools = workflow_engine.tool_registry.get_tools_by_category("weather")
    for tool in weather_tools:
        tool_name = workflow_engine.tool_registry._extract_tool_name(tool)
        if hasattr(tool, "fn"):
            is_async = inspect.iscoroutinefunction(tool.fn)
        else:
            is_async = inspect.iscoroutinefunction(tool)

        if is_async:
            print(f"    ✓ {tool_name}: async")
        else:
            print(f"    ⚠ {tool_name}: not async (may be ok)")

    # Timer tools should return dicts (intents)
    print("\n  Timer tools:")
    timer_file = Path("roles/timer/tools.py")
    if timer_file.exists():
        content = timer_file.read_text()
        if "return {" in content or "return dict(" in content:
            print("    ✓ Timer tools return dict (intent pattern)")
        else:
            print("    ⚠ Timer tools may not return intents")

    # Final report
    print("\n" + "=" * 70)
    if issues:
        print("✗ TOOL IMPLEMENTATION CHECK FAILED")
        print("=" * 70)
        print(f"\nFound {len(issues)} issues:\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        return 1
    else:
        print("✓ TOOL IMPLEMENTATION CHECK PASSED")
        print("=" * 70)
        print("\nAll tool implementations verified!")
        return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(check_tool_implementations())
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n✗ Tool implementation check failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
