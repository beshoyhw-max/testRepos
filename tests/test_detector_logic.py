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
        self.detector.static_objects.add(99)
        persons = [[100, 100, 200, 300, 1]] # H=200
        # Phone far away (wrist check fail)
        phones = [[300, 200, 310, 210, 0.9, 99]]

        mapping = self.detector._associate_phones_to_persons_advanced(persons, phones, {}, 0)
        self.assertFalse(mapping.get(1))

    def test_static_phone_in_hand(self):
        """
        Test that a static phone IS matched if held in hand (wrist close).
        """
        self.detector.static_objects.add(99)
        persons = [[100, 100, 200, 300, 1]] # H=200

        # Phone at chest (150, 200)
        phones = [[145, 195, 155, 205, 0.9, 99]]

        # Wrist very close: (140, 200). Dist = 10px.
        # 10px / 200px = 0.05. < 0.15 Threshold.
        kpts = np.zeros((17, 2))
        kpts[9] = [140, 200]
        kpts_map = {1: kpts}

        mapping = self.detector._associate_phones_to_persons_advanced(persons, phones, kpts_map, 0)
        self.assertTrue(mapping.get(1))

    def test_velocity_mismatch(self):
        """Test penalty when person moves but phone is static."""
        self.detector.velocities[1] = (3.0, 0.0)
        self.detector.velocities[99] = (0.0, 0.0)

        persons = [[100, 100, 200, 300, 1]]
        phones = [[150, 200, 160, 210, 0.9, 99]]

        mapping = self.detector._associate_phones_to_persons_advanced(persons, phones, {}, 0)
        self.assertFalse(mapping.get(1))

    def test_sticky_association(self):
        """Test that previous matches get a bonus."""
        self.detector.last_matches[1] = 99
        persons = [[100, 100, 200, 300, 1]]
        phones = [[100, 322, 110, 332, 0.9, 99]]
        mapping = self.detector._associate_phones_to_persons_advanced(persons, phones, {}, 0)
        self.assertTrue(mapping.get(1))

    def test_active_arm_elbow_angle(self):
        """Test active arm selection."""
        persons = [[100, 100, 200, 300, 1]]
        phones = [[180, 200, 190, 210, 0.9, 99]]

        kpts = np.zeros((17, 2))
        kpts[5] = [120, 120] # Left Shoulder
        kpts[7] = [120, 200] # Left Elbow
        kpts[9] = [120, 280] # Left Wrist (Straight)

        kpts[6] = [180, 120] # Right Shoulder
        kpts[8] = [180, 200] # Right Elbow
        kpts[10] = [185, 205] # Right Wrist (Bent)

        kpts_map = {1: kpts}
        mapping = self.detector._associate_phones_to_persons_advanced(persons, phones, kpts_map, 0)
        self.assertTrue(mapping.get(1))

if __name__ == '__main__':
    unittest.main()
