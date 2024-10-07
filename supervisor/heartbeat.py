import threading
import time
import logging

logger = logging.getLogger(__name__)

class Heartbeat:
    # TODO: Not used yet, need to implement support in Supervisor
    def __init__(self, supervisor, interval=5):
        self.supervisor = supervisor
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run)

    def start(self):
        self.thread.start()
        logger.info("Heartbeat started.")

    def stop(self):
        self.stop_event.set()
        logger.info("Heartbeat stopped.")

    def run(self):
        while not self.stop_event.is_set():
            self.supervisor.check_agent_status()
            self.supervisor.monitor_task_progress()
            self.supervisor.schedule_new_tasks()
            time.sleep(self.interval)
