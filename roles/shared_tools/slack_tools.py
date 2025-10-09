"""Slack tools for the Universal Agent system.

These tools provide Slack integration functionality using @tool decorated functions.
"""

import logging
import os
from datetime import datetime
from typing import Any, Optional

from strands import tool

logger = logging.getLogger(__name__)


@tool
def send_slack_message(
    channel: str, message: str, thread_ts: Optional[str] = None
) -> dict[str, Any]:
    """Send a message to a Slack channel using real Slack API.

    This tool sends messages to Slack channels using the Slack WebClient.
    Requires SLACK_BOT_TOKEN environment variable.

    Args:
        channel: Slack channel name or ID (e.g., "#general" or "C1234567890")
        message: Message content to send
        thread_ts: Optional timestamp of parent message to reply in thread

    Returns:
        Dict containing send result and metadata
    """
    logger.info(f"Sending Slack message to channel: {channel}")

    if not message or not message.strip():
        error_msg = "Message cannot be empty"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "channel": channel,
            "timestamp": datetime.now().isoformat(),
        }

    # Get required Slack tokens from environment variables
    slack_bot_token = os.getenv("SLACK_BOT_TOKEN")

    if not slack_bot_token:
        error_msg = (
            "SLACK_BOT_TOKEN environment variable is required for Slack integration"
        )
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "channel": channel,
            "timestamp": datetime.now().isoformat(),
        }

    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError

        # Initialize Slack WebClient
        client = WebClient(token=slack_bot_token)

        # Send message to Slack
        response = client.chat_postMessage(
            channel=channel, text=message, thread_ts=thread_ts
        )

        # Extract response data
        slack_response = {
            "status": "success",
            "channel": response.get("channel"),
            "message": message,
            "message_ts": response.get("ts"),
            "thread_ts": thread_ts,
            "timestamp": datetime.now().isoformat(),
            "slack_response": {
                "ok": response.get("ok"),
                "channel": response.get("channel"),
                "ts": response.get("ts"),
                "message": response.get("message", {}),
            },
        }

        logger.info(
            f"Slack message sent successfully to {channel}, ts: {response.get('ts')}"
        )
        return slack_response

    except ImportError as e:
        error_msg = f"Slack SDK not installed. Please install with: pip install slack-sdk. Error: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "channel": channel,
            "timestamp": datetime.now().isoformat(),
        }
    except SlackApiError as e:
        error_msg = f"Slack API error: {e.response['error']}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "slack_error": e.response["error"],
            "status_code": e.response.status_code,
            "channel": channel,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        error_msg = f"Failed to send Slack message: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "channel": channel,
            "timestamp": datetime.now().isoformat(),
        }


def get_slack_channel_info(channel: str) -> dict[str, Any]:
    """Get information about a Slack channel.

    Args:
        channel: Slack channel name or ID

    Returns:
        Dict containing channel information
    """
    logger.info(f"Getting Slack channel info: {channel}")

    # Mock channel info - in real implementation, this would use slack_sdk
    mock_info = {
        "channel": channel,
        "name": channel.lstrip("#"),
        "id": "C1234567890",
        "is_channel": True,
        "is_private": False,
        "member_count": 42,
        "topic": f"Discussion about {channel.lstrip('#')} topics",
        "purpose": f"Channel for {channel.lstrip('#')} related conversations",
        "timestamp": datetime.now().isoformat(),
        "mock": True,
        "status": "success",
    }

    logger.info(f"Channel info retrieved for {channel}")
    return mock_info


def list_slack_channels(
    limit: int = 20, types: str = "public_channel"
) -> dict[str, Any]:
    """List available Slack channels.

    Args:
        limit: Maximum number of channels to return
        types: Types of channels to list (public_channel, private_channel, mpim, im)

    Returns:
        Dict containing list of channels
    """
    logger.info(f"Listing Slack channels: limit={limit}, types={types}")

    # Mock channel list - in real implementation, this would use slack_sdk
    mock_channels = [
        {
            "id": "C1234567890",
            "name": "general",
            "is_private": False,
            "member_count": 100,
        },
        {
            "id": "C1234567891",
            "name": "random",
            "is_private": False,
            "member_count": 75,
        },
        {
            "id": "C1234567892",
            "name": "development",
            "is_private": False,
            "member_count": 25,
        },
        {
            "id": "C1234567893",
            "name": "announcements",
            "is_private": False,
            "member_count": 150,
        },
    ]

    # Limit results
    channels = mock_channels[:limit]

    result = {
        "channels": channels,
        "channel_count": len(channels),
        "limit": limit,
        "types": types,
        "timestamp": datetime.now().isoformat(),
        "mock": True,
        "status": "success",
    }

    logger.info(f"Listed {len(channels)} Slack channels")
    return result


def format_slack_message(content: str, formatting: str = "plain") -> dict[str, Any]:
    """Format message content for Slack.

    Args:
        content: Message content to format
        formatting: Formatting type - "plain", "markdown", "blocks"

    Returns:
        Dict containing formatted message
    """
    logger.info(
        f"Formatting Slack message: {len(content)} characters, format: {formatting}"
    )

    if not content or not content.strip():
        return {
            "formatted_message": "",
            "formatting": formatting,
            "status": "empty_input",
        }

    formatted_content = content.strip()

    if formatting == "markdown":
        # Basic markdown formatting for Slack
        formatted_content = content.replace("**", "*")  # Bold
        formatted_content = formatted_content.replace("__", "_")  # Italic
    elif formatting == "blocks":
        # Create Slack block format
        formatted_content = {
            "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": content}}]
        }

    result = {
        "formatted_message": formatted_content,
        "original_content": content,
        "formatting": formatting,
        "character_count": len(content),
        "timestamp": datetime.now().isoformat(),
        "status": "success",
    }

    logger.info(f"Message formatted successfully: {formatting} format")
    return result


def validate_slack_channel(channel: str) -> dict[str, Any]:
    """Validate Slack channel name format.

    Args:
        channel: Channel name to validate

    Returns:
        Dict containing validation results
    """
    logger.info(f"Validating Slack channel: {channel}")

    errors = []
    warnings = []
    suggestions = []

    if not channel:
        errors.append("Channel name cannot be empty")
    else:
        # Check channel format
        if not channel.startswith("#") and not channel.startswith("C"):
            warnings.append("Channel should start with # for names or C for IDs")
            suggestions.append(f"Consider using #{channel}")

        if channel.startswith("#"):
            channel_name = channel[1:]
            if len(channel_name) < 1:
                errors.append("Channel name cannot be empty after #")
            elif len(channel_name) > 21:
                errors.append("Channel name cannot be longer than 21 characters")
            elif not channel_name.replace("-", "").replace("_", "").isalnum():
                errors.append(
                    "Channel name can only contain letters, numbers, hyphens, and underscores"
                )
            elif channel_name != channel_name.lower():
                warnings.append("Channel names should be lowercase")
                suggestions.append(f"Consider using #{channel_name.lower()}")

    is_valid = len(errors) == 0

    result = {
        "channel": channel,
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "normalized_channel": channel.lower() if channel.startswith("#") else channel,
    }

    logger.info(
        f"Channel validation: {'PASSED' if is_valid else 'FAILED'} with {len(errors)} errors"
    )
    return result


def get_slack_user_info(user_id: str) -> dict[str, Any]:
    """Get information about a Slack user.

    Args:
        user_id: Slack user ID

    Returns:
        Dict containing user information
    """
    logger.info(f"Getting Slack user info: {user_id}")

    # Mock user info - in real implementation, this would use slack_sdk
    mock_user_info = {
        "user_id": user_id,
        "name": "john.doe",
        "real_name": "John Doe",
        "display_name": "John",
        "email": "john.doe@example.com",
        "is_bot": False,
        "is_admin": False,
        "timezone": "America/New_York",
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "mock": True,
    }

    logger.info(f"User info retrieved for {user_id}")
    return mock_user_info


# Tool registry for Slack tools
SLACK_TOOLS = {
    "send_slack_message": send_slack_message,
    "get_slack_channel_info": get_slack_channel_info,
    "list_slack_channels": list_slack_channels,
    "format_slack_message": format_slack_message,
    "validate_slack_channel": validate_slack_channel,
    "get_slack_user_info": get_slack_user_info,
}


def get_slack_tools() -> dict[str, Any]:
    """Get all available Slack tools."""
    return SLACK_TOOLS


def get_slack_tool_descriptions() -> dict[str, str]:
    """Get descriptions of all Slack tools."""
    return {
        "send_slack_message": "Send a message to a Slack channel",
        "get_slack_channel_info": "Get information about a Slack channel",
        "list_slack_channels": "List available Slack channels",
        "format_slack_message": "Format message content for Slack",
        "validate_slack_channel": "Validate Slack channel name format",
        "get_slack_user_info": "Get information about a Slack user",
    }
