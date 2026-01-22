import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from detector import PhoneDetector

class TestDetectorAdvancedLogic(unittest.TestCase):
    def setUp(self):
        self.yolo_patcher = patch('detector.YOLO')
        self.mock_yolo_cls = self.yolo_patcher.start()
        self.mock_yolo_instance = MagicMock()
        self.mock_yolo_cls.return_value = self.mock_yolo_instance

        self.sleep_patcher = patch('detector.SleepDetector')
        self.mock_sleep_cls = self.sleep_patcher.start()
        self.mock_sleep_instance = MagicMock()
        self.mock_sleep_cls.return_value = self.mock_sleep_instance

        self.detector = PhoneDetector(camera_id=1)

    def tearDown(self):
        self.yolo_patcher.stop()
        self.sleep_patcher.stop()

    def test_static_suppression(self):
        """Test that static phones are penalized."""
        # Setup: Mark phone ID 99 as static
        self.detector.static_objects.add(99)

        persons = [[100, 100, 200, 300, 1]] # H=200
        phones = [[150, 200, 160, 210, 0.9, 99]] # Near chest

        # Should fail match due to +5.0 penalty (unless wrist is super close)
        mapping = self.detector._associate_phones_to_persons_advanced(persons, phones, {})
        self.assertFalse(mapping.get(1))

    def test_velocity_mismatch(self):
        """Test penalty when person moves but phone is static."""
        # Person 1 moving fast (vx=3.0)
        self.detector.velocities[1] = (3.0, 0.0)
        # Phone 99 static
        self.detector.velocities[99] = (0.0, 0.0)

        persons = [[100, 100, 200, 300, 1]]
        phones = [[150, 200, 160, 210, 0.9, 99]] # Close geometrically

        # Should fail due to velocity mismatch penalty
        mapping = self.detector._associate_phones_to_persons_advanced(persons, phones, {})
        self.assertFalse(mapping.get(1))

    def test_sticky_association(self):
        """Test that previous matches get a bonus."""
        # Previous match: Person 1 -> Phone 99
        self.detector.last_matches[1] = 99

        persons = [[100, 100, 200, 300, 1]]
        # Phone slightly far away (normally might fail threshold)
        # Threshold is 0.6. Dist=0.61
        # With bonus 0.6x, it becomes 0.366 -> PASS
        phones = [[100, 322, 110, 332, 0.9, 99]] # ~122px away vertical. 122/200 = 0.61

        # Without bonus, this would fail (0.61 > 0.6)
        # With bonus, it should pass
        mapping = self.detector._associate_phones_to_persons_advanced(persons, phones, {})
        self.assertTrue(mapping.get(1))

    def test_elbow_angle_penalty(self):
        """Test penalty for straight arm (angle ~180)."""
        persons = [[100, 100, 200, 300, 1]]
        phones = [[150, 200, 160, 210, 0.9, 99]] # Perfect position

        # Mock Keypoints: Shoulder(5), Elbow(7), Wrist(9)
        # Straight line down: (150,120) -> (150,200) -> (150,280)
        kpts = np.zeros((17, 2))
        kpts[5] = [150, 120]
        kpts[7] = [150, 200]
        kpts[9] = [150, 280]

        kpts_map = {1: kpts}

        # Should fail due to straight arm penalty (+1.0 cost)
        mapping = self.detector._associate_phones_to_persons_advanced(persons, phones, kpts_map)
        self.assertFalse(mapping.get(1))

if __name__ == '__main__':
    unittest.main()
