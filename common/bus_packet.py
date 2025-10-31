"""Message bus packet definitions for the StrandsAgent Universal Agent System.

Defines data structures for inter-component communication including
task assignments, status updates, and workflow coordination messages.
"""

import uuid
from typing import Any, Optional

from pydantic import BaseModel

from common.message_bus import MessageType


class BusPacket(BaseModel):
    """Message packet for inter-component communication via the message bus.

    Encapsulates message data, routing information, and metadata for
    reliable message delivery across system components.
    """

    message_type: MessageType
    payload: Any
    sender_id: str = None
    recipient_id: str = None
    packet_id: str = "pkt_" + str(uuid.uuid4()).split("-")[-1]
    request_id: str | None = None
    metadata: dict | None = None

    def send(self, message_bus):
        """Send this packet via the provided message bus.

        Args:
            message_bus: The message bus instance to send the packet through.
        """
        message_bus.publish(self.sender_id, self.message_type, self)
