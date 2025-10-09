"""
Email channel handler for the Communication Manager.

This handler provides email notification capabilities using
either SMTP or AWS SES for delivery.
"""

import asyncio
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Union

import aiohttp
import boto3
from botocore.exceptions import ClientError

from common.communication_manager import ChannelHandler, ChannelType, MessageFormat

logger = logging.getLogger(__name__)


class EmailChannelHandler(ChannelHandler):
    """
    Channel handler for sending email notifications.

    Supports:
    - Plain text and HTML email formats
    - SMTP and AWS SES delivery methods
    - Email templates
    """

    channel_type = ChannelType.EMAIL

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize the email channel handler."""
        super().__init__(config)

        # Extract configuration
        self.delivery_method = self.config.get("delivery_method", "smtp")
        self.from_email = self.config.get("from_email", "notifications@example.com")
        self.from_name = self.config.get("from_name", "Timer Notification")

        # SMTP configuration
        self.smtp_host = self.config.get("smtp_host", "localhost")
        self.smtp_port = self.config.get("smtp_port", 587)
        self.smtp_username = self.config.get("smtp_username")
        self.smtp_password = self.config.get("smtp_password")
        self.smtp_use_tls = self.config.get("smtp_use_tls", True)

        # AWS SES configuration
        self.aws_region = self.config.get("aws_region", "us-east-1")
        self.aws_access_key = self.config.get("aws_access_key")
        self.aws_secret_key = self.config.get("aws_secret_key")

        # Validate configuration
        if self.delivery_method == "smtp" and not self.smtp_host:
            logger.warning("Email handler initialized without SMTP host")
            self.enabled = False
        elif self.delivery_method == "ses" and not (
            self.aws_access_key and self.aws_secret_key
        ):
            logger.warning("Email handler initialized without AWS credentials")
            self.enabled = False

    async def _send(
        self,
        message: str,
        recipient: Optional[str],
        message_format: MessageFormat,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Send an email notification.

        Args:
            message: The message content
            recipient: Email address to send to
            message_format: Format of the message content
            metadata: Additional metadata including:
                - subject: Email subject line
                - cc: List of CC recipients
                - bcc: List of BCC recipients
                - reply_to: Reply-to email address
                - attachments: List of attachment objects

        Returns:
            Dict with status information
        """
        if not recipient:
            return {"success": False, "error": "No recipient email address provided"}

        # Extract email metadata
        subject = metadata.get("subject", "Timer Notification")
        cc = metadata.get("cc", [])
        bcc = metadata.get("bcc", [])
        reply_to = metadata.get("reply_to", self.from_email)

        # Determine content type based on message format
        is_html = message_format in [MessageFormat.HTML, MessageFormat.RICH_TEXT]

        # Create the email message
        email_message = self._create_email_message(
            subject=subject,
            body=message,
            to_email=recipient,
            cc=cc,
            bcc=bcc,
            reply_to=reply_to,
            is_html=is_html,
        )

        # Send via the configured delivery method
        if self.delivery_method == "smtp":
            return await self._send_via_smtp(email_message, recipient)
        elif self.delivery_method == "ses":
            return await self._send_via_ses(email_message, recipient)
        else:
            return {
                "success": False,
                "error": f"Unsupported delivery method: {self.delivery_method}",
            }

    def _create_email_message(
        self,
        subject: str,
        body: str,
        to_email: str,
        cc: list[str] = None,
        bcc: list[str] = None,
        reply_to: str = None,
        is_html: bool = False,
    ) -> MIMEMultipart:
        """Create a MIME email message."""
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{self.from_name} <{self.from_email}>"
        message["To"] = to_email

        if cc:
            message["Cc"] = ", ".join(cc)

        if bcc:
            message["Bcc"] = ", ".join(bcc)

        if reply_to:
            message["Reply-To"] = reply_to

        # Attach the body with the appropriate content type
        content_type = "html" if is_html else "plain"
        message.attach(MIMEText(body, content_type))

        return message

    async def _send_via_smtp(
        self, email_message: MIMEMultipart, recipient: str
    ) -> dict[str, Any]:
        """Send an email using SMTP."""
        # Run SMTP operations in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._smtp_send, email_message, recipient)
            return {"success": True, "recipient": recipient}
        except Exception as e:
            logger.error(f"Error sending email via SMTP: {str(e)}")
            return {"success": False, "error": str(e)}

    def _smtp_send(self, email_message: MIMEMultipart, recipient: str) -> None:
        """Synchronous SMTP send operation to run in executor."""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.smtp_use_tls:
                server.starttls()

            if self.smtp_username and self.smtp_password:
                server.login(self.smtp_username, self.smtp_password)

            # Get all recipients
            all_recipients = [recipient]
            if "Cc" in email_message:
                all_recipients.extend(email_message["Cc"].split(", "))
            if "Bcc" in email_message:
                all_recipients.extend(email_message["Bcc"].split(", "))

            server.sendmail(self.from_email, all_recipients, email_message.as_string())

    async def _send_via_ses(
        self, email_message: MIMEMultipart, recipient: str
    ) -> dict[str, Any]:
        """Send an email using AWS SES."""
        # Run AWS operations in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._ses_send, email_message, recipient)
            return {"success": True, "recipient": recipient}
        except ClientError as e:
            error_message = e.response["Error"]["Message"]
            logger.error(f"Error sending email via SES: {error_message}")
            return {"success": False, "error": error_message}
        except Exception as e:
            logger.error(f"Error sending email via SES: {str(e)}")
            return {"success": False, "error": str(e)}

    def _ses_send(self, email_message: MIMEMultipart, recipient: str) -> None:
        """Synchronous AWS SES send operation to run in executor."""
        # Create SES client
        ses_client = boto3.client(
            "ses",
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
        )

        # Get all recipients
        all_recipients = [recipient]
        if "Cc" in email_message:
            all_recipients.extend(email_message["Cc"].split(", "))
        if "Bcc" in email_message:
            all_recipients.extend(email_message["Bcc"].split(", "))

        # Send the email
        ses_client.send_raw_email(
            Source=self.from_email,
            Destinations=all_recipients,
            RawMessage={"Data": email_message.as_string()},
        )
