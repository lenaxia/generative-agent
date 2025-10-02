import logging
import os
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from slack_bolt import App

class SlackBotApp:
    def __init__(self, app_token):
        self.app = App()
        self.app_token = app_token
        self.setup_middleware()
        self.setup_commands()
        self.setup_events()
        self.setup_error_handler()

    def setup_middleware(self):
        @self.app.middleware
        def log_request(logger, body, next):
            logger.debug(body)
            return next()

    def setup_commands(self):
        @self.app.command("/hello-bolt-python")
        def hello_command(ack, body):
            logger.info('Received /hello-bolt-python command')
            user_id = body["user_id"]
            ack(f"Hi <@{user_id}>!")

    def setup_events(self):
        @self.app.event("app_mention")
        def event_test(body, say, logger):
            logger.info('Received app_mention event')
            logger.info(body)
            say("What's up?")

        @self.app.event("message")
        def event_test(body, say, logger):
            logger.info('Received message event')
            logger.info(body)
            say("What's up?")

    def setup_error_handler(self):
        @self.app.error
        def global_error_handler(error, body, logger):
            logger.exception(error)
            logger.info(body)

    def run(self):
        handler = SocketModeHandler(self.app, self.app_token)
        logger.info("Starting Socket Mode Handler")
        handler.start()

        # Add additional logging for socket events
        handler.socket_mode_handler.logger.setLevel(logging.DEBUG)
        handler.socket_mode_handler.logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    app_token = os.environ["SLACK_APP_TOKEN"]
    bot_app = SlackBotApp(app_token)
    bot_app.run()
