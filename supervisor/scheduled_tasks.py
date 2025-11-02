"""Scheduled tasks for background processing.

This module provides scheduled tasks that run periodically to handle
background operations like conversation inactivity checking.
"""

import logging
import time
from typing import Any

from common.realtime_log import get_last_message_time, get_unanalyzed_messages
from roles.shared_tools.conversation_analysis import analyze_conversation

logger = logging.getLogger(__name__)


async def check_conversation_inactivity(
    user_ids: list[str], inactivity_timeout_minutes: int = 30
) -> dict[str, Any]:
    """Check for conversations needing analysis due to inactivity.

    Runs periodically to check if users have unanalyzed messages and
    their last message was more than the timeout period ago.

    Args:
        user_ids: List of user IDs to check
        inactivity_timeout_minutes: Minutes of inactivity before triggering analysis

    Returns:
        Dict with analysis results for each user
    """
    results = {"success": True, "users_analyzed": 0, "errors": 0, "details": []}

    timeout_seconds = inactivity_timeout_minutes * 60
    current_time = time.time()

    for user_id in user_ids:
        try:
            # Check if user has unanalyzed messages
            unanalyzed = get_unanalyzed_messages(user_id)
            if not unanalyzed:
                logger.debug(f"No unanalyzed messages for user {user_id}")
                continue

            # Check last message time
            last_message_time = get_last_message_time(user_id)
            if last_message_time is None:
                logger.debug(f"No last message time for user {user_id}")
                continue

            # Calculate inactivity duration
            inactivity_duration = current_time - last_message_time

            # Trigger analysis if inactive for longer than timeout
            if inactivity_duration >= timeout_seconds:
                logger.info(
                    f"Triggering analysis for user {user_id} "
                    f"(inactive for {inactivity_duration / 60:.1f} minutes)"
                )

                analysis_result = await analyze_conversation(user_id=user_id)

                if analysis_result.get("success"):
                    results["users_analyzed"] += 1
                    results["details"].append(
                        {
                            "user_id": user_id,
                            "status": "analyzed",
                            "analyzed_count": analysis_result.get("analyzed_count", 0),
                            "memories_created": analysis_result.get(
                                "memories_created", 0
                            ),
                        }
                    )
                else:
                    results["errors"] += 1
                    results["details"].append(
                        {
                            "user_id": user_id,
                            "status": "failed",
                            "error": analysis_result.get("error", "Unknown error"),
                        }
                    )
            else:
                logger.debug(
                    f"User {user_id} not inactive long enough "
                    f"({inactivity_duration / 60:.1f} < {inactivity_timeout_minutes} minutes)"
                )

        except Exception as e:
            logger.error(f"Error checking inactivity for user {user_id}: {e}")
            results["errors"] += 1
            results["details"].append(
                {"user_id": user_id, "status": "error", "error": str(e)}
            )

    logger.info(
        f"Inactivity check complete: {results['users_analyzed']} users analyzed, "
        f"{results['errors']} errors"
    )

    return results
