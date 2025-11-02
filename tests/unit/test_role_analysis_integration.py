"""Tests for role integration with conversation analysis tool.

This module tests that multi-turn roles (conversation, calendar, planning)
have the conversation_analysis tool available.
"""

from unittest.mock import MagicMock, patch

import pytest


def test_conversation_role_has_analysis_tool():
    """Test conversation role includes conversation analysis tool."""
    from roles.core_conversation import ROLE_CONFIG

    # Verify conversation_analysis is in shared tools
    shared_tools = ROLE_CONFIG.get("tools", {}).get("shared", [])
    assert (
        "conversation_analysis" in shared_tools
    ), "conversation_analysis tool should be in conversation role shared tools"


def test_calendar_role_has_analysis_tool():
    """Test calendar role includes conversation analysis tool."""
    from roles.core_calendar import ROLE_CONFIG

    # Verify conversation_analysis is in shared tools
    shared_tools = ROLE_CONFIG.get("tools", {}).get("shared", [])
    assert (
        "conversation_analysis" in shared_tools
    ), "conversation_analysis tool should be in calendar role shared tools"


def test_planning_role_has_analysis_tool():
    """Test planning role includes conversation analysis tool."""
    from roles.core_planning import ROLE_CONFIG

    # Verify conversation_analysis is in shared tools
    shared_tools = ROLE_CONFIG.get("tools", {}).get("shared", [])
    assert (
        "conversation_analysis" in shared_tools
    ), "conversation_analysis tool should be in planning role shared tools"
