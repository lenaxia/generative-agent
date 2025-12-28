"""Check tool name mismatch"""

import asyncio
import logging

logging.basicConfig(level=logging.WARNING)


async def check_tool_names():
    from common.message_bus import MessageBus
    from llm_provider.factory import LLMFactory
    from supervisor.workflow_engine import WorkflowEngine

    llm_factory = LLMFactory({})
    message_bus = MessageBus()
    workflow_engine = WorkflowEngine(llm_factory=llm_factory, message_bus=message_bus)
    await workflow_engine.initialize_phase3_systems()

    print("=" * 60)
    print("Tool Name Analysis")
    print("=" * 60)

    # Check what weather tools are registered
    print("\nWeather tools in ToolRegistry:")
    weather_tools = workflow_engine.tool_registry.get_tools_by_category("weather")
    for tool in weather_tools:
        tool_name = workflow_engine.tool_registry._extract_tool_name(tool)
        print(f"  - weather.{tool_name}")

    # Check what the weather role requires
    weather_role = workflow_engine.role_registry.get_domain_role("weather")
    print(f"\nWeather role REQUIRED_TOOLS:")
    for tool_name in weather_role.REQUIRED_TOOLS:
        print(f"  - {tool_name}")
        tool = workflow_engine.tool_registry.get_tool(tool_name)
        if tool:
            print(f"    ✓ Found")
        else:
            print(f"    ✗ NOT FOUND")

    print("\n" + "=" * 60)
    print("Checking all roles:")
    print("=" * 60)

    for role_name in ["weather", "calendar", "timer", "smart_home"]:
        role = workflow_engine.role_registry.get_domain_role(role_name)
        if role:
            print(f"\n{role_name}:")
            print(f"  Required: {len(role.REQUIRED_TOOLS)} tools")
            print(f"  Loaded:   {len(role.tools)} tools")
            if len(role.tools) != len(role.REQUIRED_TOOLS):
                print(f"  ⚠ MISMATCH!")
                for req_tool in role.REQUIRED_TOOLS:
                    tool = workflow_engine.tool_registry.get_tool(req_tool)
                    status = "✓" if tool else "✗"
                    print(f"    {status} {req_tool}")


asyncio.run(check_tool_names())
