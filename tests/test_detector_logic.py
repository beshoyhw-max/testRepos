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
        mapping = self.detector._associate_phones_to_persons_advanced(persons, phones, {}, 0)
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
        mapping = self.detector._associate_phones_to_persons_advanced(persons, phones, {}, 0)
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
        mapping = self.detector._associate_phones_to_persons_advanced(persons, phones, {}, 0)
        self.assertTrue(mapping.get(1))

    def test_active_arm_elbow_angle(self):
        """
        Test that elbow angle validation correctly picks the active arm.
        Scenario: Left arm straight (distractor), Right arm active (texting).
        Should PASS because right arm is closer to phone.
        """
        persons = [[100, 100, 200, 300, 1]]
        # Phone on Right side (closer to right arm)
        phones = [[180, 200, 190, 210, 0.9, 99]]

        # Mock Keypoints
        kpts = np.zeros((17, 2))

        # Left Arm: Straight down (150 degrees) - would fail if checked
        kpts[5] = [120, 120] # Shoulder
        kpts[7] = [120, 200] # Elbow
        kpts[9] = [120, 280] # Wrist (Far from phone)

        # Right Arm: Texting (90 degrees) - should be checked
        kpts[6] = [180, 120] # Shoulder
        kpts[8] = [180, 200] # Elbow
        kpts[10] = [185, 205] # Wrist (Close to phone)

        kpts_map = {1: kpts}

        mapping = self.detector._associate_phones_to_persons_advanced(persons, phones, kpts_map, 0)
        self.assertTrue(mapping.get(1))

if __name__ == '__main__':
    unittest.main()
