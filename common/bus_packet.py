import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel

from common.message_bus import MessageType


class BusPacket(BaseModel):
    message_type: MessageType
    payload: Any
    sender_id: str = None
    recipient_id: str = None
    packet_id: str = "pkt_" + str(uuid.uuid4()).split("-")[-1]
    request_id: Optional[str] = None
    callback_details: Optional[Dict] = None
    metadata: Optional[Dict] = None

    def send(self, message_bus):
        message_bus.publish(self.sender_id, self.message_type, self)
