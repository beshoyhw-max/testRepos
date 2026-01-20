import unittest
from unittest.mock import patch
from cooldown_manager import CooldownManager

class TestCooldownManager(unittest.TestCase):
    def setUp(self):
        self.manager = CooldownManager(cooldown_seconds=120, spatial_radius=100)

    @patch('time.time')
    def test_basic_cooldown(self, mock_time):
        mock_time.return_value = 1000

        # First capture allowed
        self.assertTrue(self.manager.should_capture(100, 100, "phone"))
        self.manager.record_capture(100, 100, "phone")

        # Second capture denied immediately
        self.assertFalse(self.manager.should_capture(100, 100, "phone"))

        # Capture allowed after 121 seconds
        mock_time.return_value = 1121
        self.assertTrue(self.manager.should_capture(100, 100, "phone"))

    @patch('time.time')
    def test_different_types(self, mock_time):
        mock_time.return_value = 1000

        # Phone capture recorded
        self.manager.record_capture(100, 100, "phone")

        # Sleep capture allowed at same location immediately
        self.assertTrue(self.manager.should_capture(100, 100, "sleep"))
        self.manager.record_capture(100, 100, "sleep")

        # Sleep capture denied again
        self.assertFalse(self.manager.should_capture(100, 100, "sleep"))

    @patch('time.time')
    def test_spatial_logic(self, mock_time):
        mock_time.return_value = 1000

        # Person A recorded at 100, 100
        self.manager.record_capture(100, 100, "phone")

        # Person A (slight movement, within 100px) denied
        self.assertFalse(self.manager.should_capture(150, 150, "phone")) # dist ~70

        # Person B (far away) allowed
        self.assertTrue(self.manager.should_capture(300, 300, "phone"))

    @patch('time.time')
    def test_cleanup(self, mock_time):
        mock_time.return_value = 1000
        self.manager.record_capture(100, 100, "phone")
        self.assertEqual(len(self.manager.history), 1)

        # Advance time past cooldown
        mock_time.return_value = 1130

        # Trigger cleanup via should_capture
        self.manager.should_capture(100, 100, "phone")
        self.assertEqual(len(self.manager.history), 0)

if __name__ == '__main__':
    unittest.main()
