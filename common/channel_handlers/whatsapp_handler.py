"""
WhatsApp channel handler for the Communication Manager.

This handler provides bidirectional integration with WhatsApp
using the WhatsApp Business API or Twilio API for messaging.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp

from common.communication_manager import ChannelHandler, ChannelType, MessageFormat

logger = logging.getLogger(__name__)


class WhatsAppChannelHandler(ChannelHandler):
    """
    Channel handler for WhatsApp messaging.

    Supports:
    - Text messaging via WhatsApp Business API
    - Media sharing (images, documents)
    - Bidirectional communication via webhooks
    - Message templates and formatting
    - Contact management
    """

    channel_type = ChannelType.WHATSAPP

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize the WhatsApp channel handler."""
        super().__init__(config)

        # Configuration
        self.api_provider = self.config.get(
            "api_provider", "twilio"
        )  # twilio, meta, 360dialog
        self.phone_number = os.environ.get("WHATSAPP_PHONE_NUMBER") or self.config.get(
            "phone_number"
        )

        # Twilio configuration
        self.twilio_account_sid = os.environ.get(
            "TWILIO_ACCOUNT_SID"
        ) or self.config.get("twilio_account_sid")
        self.twilio_auth_token = os.environ.get("TWILIO_AUTH_TOKEN") or self.config.get(
            "twilio_auth_token"
        )

        # Meta WhatsApp Business API configuration
        self.meta_access_token = os.environ.get(
            "WHATSAPP_ACCESS_TOKEN"
        ) or self.config.get("meta_access_token")
        self.meta_phone_number_id = os.environ.get(
            "WHATSAPP_PHONE_NUMBER_ID"
        ) or self.config.get("meta_phone_number_id")

        # Webhook configuration for receiving messages
        self.webhook_verify_token = os.environ.get(
            "WHATSAPP_WEBHOOK_VERIFY_TOKEN"
        ) or self.config.get("webhook_verify_token")
        self.webhook_url = self.config.get("webhook_url")

        # Runtime state
        self.pending_questions = {}
        self.contact_cache = {}

    def _validate_requirements(self) -> bool:
        """Validate WhatsApp configuration."""
        if not self.phone_number:
            logger.error(
                "WHATSAPP_PHONE_NUMBER environment variable or phone_number config required"
            )
            return False

        if self.api_provider == "twilio":
            if not (self.twilio_account_sid and self.twilio_auth_token):
                logger.error(
                    "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN required for Twilio provider"
                )
                return False
        elif self.api_provider == "meta":
            if not (self.meta_access_token and self.meta_phone_number_id):
                logger.error(
                    "WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID required for Meta provider"
                )
                return False
        else:
            logger.error(f"Unsupported WhatsApp API provider: {self.api_provider}")
            return False

        return True

    def _get_requirements_error_message(self) -> str:
        """Get descriptive error message for missing WhatsApp requirements."""
        missing = []

        if not self.phone_number:
            missing.append("WHATSAPP_PHONE_NUMBER environment variable")

        if self.api_provider == "twilio":
            if not self.twilio_account_sid:
                missing.append("TWILIO_ACCOUNT_SID environment variable")
            if not self.twilio_auth_token:
                missing.append("TWILIO_AUTH_TOKEN environment variable")
        elif self.api_provider == "meta":
            if not self.meta_access_token:
                missing.append("WHATSAPP_ACCESS_TOKEN environment variable")
            if not self.meta_phone_number_id:
                missing.append("WHATSAPP_PHONE_NUMBER_ID environment variable")

        if missing:
            return f"missing: {', '.join(missing)}"
        else:
            return f"unsupported API provider: {self.api_provider}"

    def get_capabilities(self) -> dict[str, Any]:
        """WhatsApp channel capabilities."""
        return {
            "supports_rich_text": True,  # Markdown formatting
            "supports_buttons": True,  # Interactive buttons (limited)
            "supports_audio": False,  # Voice messages not implemented yet
            "supports_images": True,  # Media sharing
            "bidirectional": True,  # Full bidirectional messaging
            "requires_session": False,  # Stateless API calls
            "max_message_length": 4096,  # WhatsApp message limit
        }

    async def _send(
        self,
        message: str,
        recipient: Optional[str],
        message_format: MessageFormat,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Send message via WhatsApp.

        Args:
            message: The message content
            recipient: Phone number (with country code, e.g., +1234567890)
            message_format: Format of the message content
            metadata: Additional options:
                - media_url: URL to media file (image, document)
                - media_type: Type of media (image, document, audio)
                - template_name: WhatsApp template name
                - template_params: Template parameters
                - buttons: Interactive buttons (limited support)

        Returns:
            Dict with status information
        """
        if not recipient:
            return {"success": False, "error": "Recipient phone number required"}

        # Clean phone number format
        recipient = self._format_phone_number(recipient)

        try:
            if self.api_provider == "twilio":
                return await self._send_via_twilio(
                    message, recipient, message_format, metadata
                )
            elif self.api_provider == "meta":
                return await self._send_via_meta(
                    message, recipient, message_format, metadata
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported provider: {self.api_provider}",
                }

        except Exception as e:
            logger.error(f"WhatsApp message send failed: {e}")
            return {"success": False, "error": str(e)}

    def _format_phone_number(self, phone: str) -> str:
        """Format phone number for WhatsApp (ensure + prefix)."""
        phone = phone.strip().replace(" ", "").replace("-", "")
        if not phone.startswith("+"):
            phone = "+" + phone
        return phone

    async def _send_via_twilio(
        self,
        message: str,
        recipient: str,
        message_format: MessageFormat,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Send message via Twilio WhatsApp API."""
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/Messages.json"

        # Prepare message data
        data = {
            "From": f"whatsapp:{self.phone_number}",
            "To": f"whatsapp:{recipient}",
            "Body": message,
        }

        # Add media if provided
        if "media_url" in metadata:
            data["MediaUrl"] = metadata["media_url"]

        # Basic auth for Twilio
        auth = aiohttp.BasicAuth(self.twilio_account_sid, self.twilio_auth_token)

        async with aiohttp.ClientSession(auth=auth) as session:
            async with session.post(url, data=data) as response:
                if response.status == 201:
                    result = await response.json()
                    return {
                        "success": True,
                        "message_sid": result.get("sid"),
                        "recipient": recipient,
                        "provider": "twilio",
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"Twilio API error: {response.status} - {error_text}",
                    }

    async def _send_via_meta(
        self,
        message: str,
        recipient: str,
        message_format: MessageFormat,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Send message via Meta WhatsApp Business API."""
        url = f"https://graph.facebook.com/v18.0/{self.meta_phone_number_id}/messages"

        headers = {
            "Authorization": f"Bearer {self.meta_access_token}",
            "Content-Type": "application/json",
        }

        # Prepare message payload
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient.replace("+", ""),  # Remove + for Meta API
            "type": "text",
            "text": {"body": message},
        }

        # Add interactive buttons if provided
        buttons = metadata.get("buttons", [])
        if buttons and len(buttons) <= 3:  # WhatsApp limit
            payload["type"] = "interactive"
            payload["interactive"] = {
                "type": "button",
                "body": {"text": message},
                "action": {
                    "buttons": [
                        {
                            "type": "reply",
                            "reply": {
                                "id": f"btn_{i}",
                                "title": btn.get("text", f"Option {i+1}")[
                                    :20
                                ],  # 20 char limit
                            },
                        }
                        for i, btn in enumerate(buttons[:3])
                    ]
                },
            }
            del payload["text"]  # Remove text when using interactive

        # Add media if provided
        if "media_url" in metadata:
            media_type = metadata.get("media_type", "image")
            payload["type"] = media_type
            payload[media_type] = {"link": metadata["media_url"]}
            if message:  # Add caption if message provided
                payload[media_type]["caption"] = message
            if "text" in payload:
                del payload["text"]

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "message_id": result.get("messages", [{}])[0].get("id"),
                        "recipient": recipient,
                        "provider": "meta",
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"Meta WhatsApp API error: {response.status} - {error_text}",
                    }

    async def _ask_question_impl(
        self, question: str, options: list[str], timeout: int
    ) -> str:
        """Ask question via WhatsApp with interactive buttons."""
        if not self.get_capabilities().get("bidirectional", False):
            raise NotImplementedError(
                "WhatsApp bidirectional communication not available"
            )

        # Generate unique question ID
        import uuid

        question_id = f"question_{uuid.uuid4().hex[:8]}"

        try:
            # Create buttons for options (max 3 for WhatsApp)
            buttons = []
            for i, option in enumerate((options or ["Yes", "No"])[:3]):
                buttons.append({"text": option, "value": option})

            # Set up response tracking
            response_future = asyncio.Future()
            self.pending_questions[question_id] = {
                "question": question,
                "options": options,
                "response_future": response_future,
            }

            # Send question with buttons
            # Note: recipient would need to be determined from context
            # For now, this is a placeholder implementation
            result = await self._send(
                question,
                self.phone_number,  # Placeholder - would use actual recipient
                MessageFormat.PLAIN_TEXT,
                {"buttons": buttons},
            )

            if not result.get("success"):
                raise Exception(f"Failed to send question: {result.get('error')}")

            # Wait for response with timeout
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            if question_id in self.pending_questions:
                del self.pending_questions[question_id]
            return "timeout"
        except Exception as e:
            if question_id in self.pending_questions:
                del self.pending_questions[question_id]
            logger.error(f"WhatsApp question failed: {e}")
            raise

    async def handle_incoming_webhook(
        self, webhook_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle incoming WhatsApp webhook (for bidirectional communication)."""
        try:
            if self.api_provider == "twilio":
                return await self._handle_twilio_webhook(webhook_data)
            elif self.api_provider == "meta":
                return await self._handle_meta_webhook(webhook_data)
            else:
                return {"success": False, "error": "Unsupported provider for webhooks"}

        except Exception as e:
            logger.error(f"Webhook handling failed: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_twilio_webhook(
        self, webhook_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle Twilio WhatsApp webhook."""
        from_number = webhook_data.get("From", "").replace("whatsapp:", "")
        message_body = webhook_data.get("Body", "")
        message_sid = webhook_data.get("MessageSid")

        if message_body and self.message_queue:
            await self.message_queue.put(
                {
                    "type": "incoming_message",
                    "user_id": from_number,
                    "channel_id": "whatsapp",
                    "text": message_body,
                    "timestamp": webhook_data.get("DateCreated"),
                    "metadata": {"message_sid": message_sid, "provider": "twilio"},
                }
            )

        return {"success": True, "processed": True}

    async def _handle_meta_webhook(
        self, webhook_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle Meta WhatsApp Business API webhook."""
        entry = webhook_data.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})

        # Handle incoming messages
        messages = value.get("messages", [])
        for message in messages:
            from_number = message.get("from")
            message_type = message.get("type")
            message_id = message.get("id")

            text_content = ""
            if message_type == "text":
                text_content = message.get("text", {}).get("body", "")
            elif message_type == "interactive":
                # Handle button responses
                interactive = message.get("interactive", {})
                if interactive.get("type") == "button_reply":
                    button_reply = interactive.get("button_reply", {})
                    text_content = button_reply.get("title", "")

                    # Check if this is a response to a pending question
                    button_id = button_reply.get("id", "")
                    for question_id, question_data in self.pending_questions.items():
                        if text_content in question_data.get("options", []):
                            question_data["response_future"].set_result(text_content)
                            del self.pending_questions[question_id]
                            break

            if text_content and self.message_queue:
                await self.message_queue.put(
                    {
                        "type": "incoming_message",
                        "user_id": from_number,
                        "channel_id": "whatsapp",
                        "text": text_content,
                        "timestamp": message.get("timestamp"),
                        "metadata": {
                            "message_id": message_id,
                            "message_type": message_type,
                            "provider": "meta",
                        },
                    }
                )

        return {"success": True, "processed": len(messages)}

    async def send_media(
        self,
        media_url: str,
        recipient: str,
        caption: str = "",
        media_type: str = "image",
    ) -> dict[str, Any]:
        """Send media message via WhatsApp."""
        metadata = {"media_url": media_url, "media_type": media_type}

        return await self._send(caption, recipient, MessageFormat.PLAIN_TEXT, metadata)

    async def send_template(
        self, template_name: str, recipient: str, parameters: list[str] = None
    ) -> dict[str, Any]:
        """Send WhatsApp template message."""
        if self.api_provider != "meta":
            return {"success": False, "error": "Templates only supported with Meta API"}

        url = f"https://graph.facebook.com/v18.0/{self.meta_phone_number_id}/messages"

        headers = {
            "Authorization": f"Bearer {self.meta_access_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "messaging_product": "whatsapp",
            "to": recipient.replace("+", ""),
            "type": "template",
            "template": {"name": template_name, "language": {"code": "en_US"}},
        }

        # Add parameters if provided
        if parameters:
            payload["template"]["components"] = [
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": param} for param in parameters
                    ],
                }
            ]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "message_id": result.get("messages", [{}])[0].get("id"),
                            "template": template_name,
                            "recipient": recipient,
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Template send failed: {response.status} - {error_text}",
                        }

        except Exception as e:
            logger.error(f"Template send error: {e}")
            return {"success": False, "error": str(e)}

    async def get_contact_info(self, phone_number: str) -> dict[str, Any]:
        """Get contact information from WhatsApp."""
        if self.api_provider != "meta":
            return {
                "success": False,
                "error": "Contact info only available with Meta API",
            }

        # Check cache first
        if phone_number in self.contact_cache:
            return {"success": True, "contact": self.contact_cache[phone_number]}

        # This would require additional API calls to get contact info
        # For now, return basic info
        contact_info = {
            "phone_number": phone_number,
            "name": f"Contact {phone_number[-4:]}",  # Last 4 digits
            "profile_name": None,
        }

        self.contact_cache[phone_number] = contact_info
        return {"success": True, "contact": contact_info}

    async def _ask_question_impl(
        self, question: str, options: list[str], timeout: int
    ) -> str:
        """Ask question via WhatsApp with interactive buttons."""
        if not self.get_capabilities().get("bidirectional", False):
            raise NotImplementedError(
                "WhatsApp bidirectional communication not available"
            )

        # Generate unique question ID
        import uuid

        question_id = f"question_{uuid.uuid4().hex[:8]}"

        try:
            # Create buttons for options (max 3 for WhatsApp)
            buttons = []
            for option in (options or ["Yes", "No"])[:3]:
                buttons.append({"text": option, "value": option})

            # Set up response tracking
            response_future = asyncio.Future()
            self.pending_questions[question_id] = {
                "question": question,
                "options": options,
                "response_future": response_future,
            }

            # Send question with buttons
            # Note: This would need a recipient from context
            # For now, this is a placeholder implementation
            result = await self._send(
                question,
                self.phone_number,  # Placeholder - would use actual recipient
                MessageFormat.PLAIN_TEXT,
                {"buttons": buttons},
            )

            if not result.get("success"):
                raise Exception(f"Failed to send question: {result.get('error')}")

            # Wait for response with timeout
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            if question_id in self.pending_questions:
                del self.pending_questions[question_id]
            return "timeout"
        except Exception as e:
            if question_id in self.pending_questions:
                del self.pending_questions[question_id]
            logger.error(f"WhatsApp question failed: {e}")
            raise
