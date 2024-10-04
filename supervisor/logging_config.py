import logging
import json
import re

# Define regular expressions to match your credentials
credential_patterns = [
    r"AWS_ACCESS_KEY_ID='?\S*'?",
    r"AWS_SECRET_ACCESS_KEY='?\S*'?",
    r"'aws_access_key_id': '?\S*'?",
    r"'aws_secret_access_key': '?\S*'?",
    # Add more patterns as needed
    # Add more patterns as needed
]

def configure_logging(log_level=logging.INFO, log_file=None):
    log_level = logging.getLevelName(log_level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)] if log_file else [logging.StreamHandler()],
    )

    # Create a logger for the Supervisor class
    supervisor_logger = logging.getLogger("supervisor")

    credential_filter = CredentialFilter(credential_patterns)
    supervisor_logger.addFilter(credential_filter)

class CredentialFilter(logging.Filter):
    def __init__(self, credential_patterns):
        self.credential_patterns = credential_patterns
        super().__init__()

    def filter(self, record):
        message = record.getMessage()
        try:
            # Try to parse the message as a dictionary
            message_dict = json.loads(message)
            for key, value in message_dict.items():
                for pattern in self.credential_patterns:
                    if re.match(pattern, str(value)):
                        message_dict[key] = 'REDACTED'
            record.msg = json.dumps(message_dict)
        except (ValueError, TypeError):
            # If the message is not a dictionary, treat it as a regular string
            for pattern in self.credential_patterns:
                for match in re.finditer(pattern, message):
                    value = match.group()
                    if value != 'None':
                        message = message.replace(value, 'REDACTED')
            record.msg = message
        return True
