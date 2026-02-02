import unittest
from unittest.mock import patch
from cooldown_manager import CooldownManager

class TestCooldownManager(unittest.TestCase):
    def setUp(self):
        self.manager = CooldownManager(cooldown_seconds=120, spatial_radius=100)

    @patch('time.time')
    def test_id_based_cooldown(self, mock_time):
        mock_time.return_value = 1000

        # Capture ID 1 (Person A) - Phone
        self.assertTrue(self.manager.should_capture(1, 100, 100, "phone"))
        self.manager.record_capture(1, 100, 100, "phone")

        # ID 1 again immediately -> Block
        self.assertFalse(self.manager.should_capture(1, 120, 120, "phone"))

        # ID 2 (Person B) at same location -> Allow
        self.assertTrue(self.manager.should_capture(2, 100, 100, "phone"))

        # Advance time
        mock_time.return_value = 1121
        # ID 1 again -> Allow
        self.assertTrue(self.manager.should_capture(1, 100, 100, "phone"))

    @patch('time.time')
    def test_spatial_fallback(self, mock_time):
        mock_time.return_value = 1000

        # Capture ID 1 at 100,100
        self.manager.record_capture(1, 100, 100, "phone")

        # Tracker fails (ID=None), but location is close (105, 105) -> Block
        self.assertFalse(self.manager.should_capture(None, 105, 105, "phone"))

        # Tracker fails (ID=None), location far (300, 300) -> Allow
        self.assertTrue(self.manager.should_capture(None, 300, 300, "phone"))

    @patch('time.time')
    def test_different_types_with_id(self, mock_time):
        mock_time.return_value = 1000

        # ID 1 Phone
        self.manager.record_capture(1, 100, 100, "phone")

        # ID 1 Sleep -> Allow (different type)
        self.assertTrue(self.manager.should_capture(1, 100, 100, "sleep"))
        self.manager.record_capture(1, 100, 100, "sleep")

        # ID 1 Sleep again -> Block
        self.assertFalse(self.manager.should_capture(1, 100, 100, "sleep"))

if __name__ == '__main__':
    unittest.main()
