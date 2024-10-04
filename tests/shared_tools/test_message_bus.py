import unittest
from unittest.mock import Mock, call
from shared_tools import MessageBus, MessageType

class TestMessageBus(unittest.TestCase):
    def setUp(self):
        self.message_bus = MessageBus()

    def test_start_stop(self):
        self.assertFalse(self.message_bus.is_running())
        self.message_bus.start()
        self.assertTrue(self.message_bus.is_running())
        self.message_bus.stop()
        self.assertFalse(self.message_bus.is_running())

    def test_subscribe_unsubscribe(self):
        subscriber = object()
        callback = Mock()

        self.message_bus.subscribe(subscriber, MessageType.TASK_ASSIGNMENT, callback)
        self.assertIn(subscriber, self.message_bus._subscribers[MessageType.TASK_ASSIGNMENT])
        self.assertIn(callback, self.message_bus._subscribers[MessageType.TASK_ASSIGNMENT][subscriber])

        self.message_bus.unsubscribe(subscriber, MessageType.TASK_ASSIGNMENT, callback)
        self.assertNotIn(callback, self.message_bus._subscribers[MessageType.TASK_ASSIGNMENT][subscriber])

        self.message_bus.unsubscribe(subscriber, MessageType.TASK_ASSIGNMENT)
        self.assertNotIn(subscriber, self.message_bus._subscribers[MessageType.TASK_ASSIGNMENT])

    def test_publish_subscribe(self):
        subscriber1 = object()
        subscriber2 = object()
        callback1 = Mock()
        callback2 = Mock()

        self.message_bus.start()
        self.message_bus.subscribe(subscriber1, MessageType.TASK_ASSIGNMENT, callback1)
        self.message_bus.subscribe(subscriber2, MessageType.TASK_ASSIGNMENT, callback2)

        message = "Hello, world!"
        self.message_bus.publish(None, MessageType.TASK_ASSIGNMENT, message)
        callback1.assert_called_once_with(message)
        callback2.assert_called_once_with(message)

        callback1.reset_mock()
        callback2.reset_mock()
        self.message_bus.stop()
        self.message_bus.publish(None, MessageType.TASK_ASSIGNMENT, message)
        callback1.assert_not_called()
        callback2.assert_not_called()

    def test_publish_unsubscribed(self):
        subscriber = object()
        callback = Mock()

        self.message_bus.start()
        self.message_bus.subscribe(subscriber, MessageType.TASK_ASSIGNMENT, callback)
        self.message_bus.unsubscribe(subscriber, MessageType.TASK_ASSIGNMENT)

        message = "Hello, world!"
        self.message_bus.publish(None, MessageType.TASK_ASSIGNMENT, message)
        callback.assert_not_called()

if __name__ == '__main__':
    unittest.main()
