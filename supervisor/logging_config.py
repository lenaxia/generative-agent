"""Logging configuration module for the supervisor system.

This module provides centralized logging configuration, formatters, and
handlers for the supervisor and its components, ensuring consistent
logging across the entire system.
"""

import json
import logging
import os
import re
from inspect import getmodulename

from supervisor.supervisor_config import LoggingConfig

# Define regular expressions to match your credentials
credential_patterns = [
    r"AWS_ACCESS_KEY_ID='?\S*'?",
    r"AWS_SECRET_ACCESS_KEY='?\S*'?",
    r"'aws_access_key_id': '?\S*'?",
    r"'aws_secret_access_key': '?\S*'?",
    # Add more patterns as needed
]


class CustomLogRecord(logging.LogRecord):
    """Custom log record with enhanced module name tracking.

    Extends the standard LogRecord to include module name information
    for improved logging context and debugging capabilities.
    """

    def __init__(self, *args, **kwargs):
        """Initialize CustomLogRecord with module name extraction.

        Args:
            *args: Positional arguments passed to LogRecord.
            **kwargs: Keyword arguments passed to LogRecord.
        """
        super().__init__(*args, **kwargs)
        self.module_name = getmodulename(self.pathname)


class ModuleNameFormatter(logging.Formatter):
    """Custom log formatter with module name support.

    Provides enhanced log formatting capabilities with support for
    module name display and custom formatting patterns.
    """

    def format(self, record):
        """Format a log record using the parent formatter.

        Applies the standard formatting to the log record while maintaining
        compatibility with the custom module name functionality.

        Args:
            record: The LogRecord instance to format.

        Returns:
            The formatted log message string.
        """
        return super().format(record)


def configure_logging(logging_config: LoggingConfig):
    """Configure logging based on the provided logging configuration.

    Args:
        logging_config: Configuration object containing logging settings.
    """
    log_level = logging.getLevelName(logging_config.log_level.upper())
    log_file = logging_config.log_file
    disable_console_logging = logging_config.disable_console_logging
    log_file_max_size = (
        logging_config.log_file_max_size * 1024 * 1024
    )  # Convert from MB to bytes

    logging.setLogRecordFactory(CustomLogRecord)  # Set the custom LogRecord class
    handlers = []

    if log_file:
        # Strip leading and trailing whitespace and slashes
        log_file = log_file.strip().strip("/")

        # Replace multiple consecutive slashes with a single slash
        log_file = re.sub(r"/+", "/", log_file)

        # Convert relative path to absolute path
        if not os.path.isabs(log_file):
            log_file = os.path.join(os.getcwd(), log_file)

        # Create the directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Check if the log file exists and create it if it doesn't
        if not os.path.exists(log_file):
            open(log_file, "w").close()

        file_handler = logging.FileHandler(log_file)

        # Check if the log file size exceeds the maximum size
        if os.path.getsize(log_file) > log_file_max_size:
            with open(log_file, "r+") as f:
                data = f.read()
                f.seek(0)
                f.write(data[len(data) - log_file_max_size :])
                f.truncate()

        handlers.append(file_handler)

    if not disable_console_logging:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [%(module_name)s] %(message)s",
        handlers=handlers,
    )

    # Create a logger for the Supervisor class
    supervisor_logger = logging.getLogger("supervisor")

    # Add a custom formatter to include the module name
    formatter = ModuleNameFormatter()
    for handler in supervisor_logger.handlers:
        handler.setFormatter(formatter)

    # credential_filter = CredentialFilter(credential_patterns)
    # supervisor_logger.addFilter(credential_filter)


class CredentialFilter(logging.Filter):
    """Logging filter for credential sanitization.

    Filters and sanitizes log messages to prevent credential leakage
    by detecting and masking sensitive information patterns.

    Note: This implementation is currently under development.
    """

    def __init__(self, credential_patterns):
        """Initialize CredentialFilter with credential patterns.

        Args:
            credential_patterns: Patterns to detect and filter credentials.
        """
        self.credential_patterns = credential_patterns
        super().__init__()

    def filter(self, record):
        """Filter and sanitize log records to prevent credential leakage.

        Examines log messages for credential patterns and replaces any
        matching sensitive information with "REDACTED" to prevent
        accidental exposure of secrets in logs.

        Args:
            record: The LogRecord instance to filter and sanitize.

        Returns:
            bool: Always returns True to allow the record to be logged
                  after sanitization.
        """
        message = record.getMessage()
        try:
            # Try to parse the message as a dictionary
            message_dict = json.loads(message)
            for key, value in message_dict.items():
                for pattern in self.credential_patterns:
                    if re.match(pattern, str(value)):
                        message_dict[key] = "REDACTED"
            record.msg = json.dumps(message_dict)
        except (ValueError, TypeError):
            # If the message is not a dictionary, treat it as a regular string
            for pattern in self.credential_patterns:
                for match in re.finditer(pattern, message):
                    value = match.group()
                    if value != "None":
                        message = message.replace(value, "REDACTED")
            record.msg = message
        return True
