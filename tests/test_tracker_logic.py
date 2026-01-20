import unittest
from unittest.mock import MagicMock, patch
import time
import sys

# Mock dependencies
sys.modules['cv2'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
# We DO NOT mock 'detector' because we want to test it
sys.modules['sleep_detector'] = MagicMock()

# Now import the class to test
from detector import PhoneDetector

class TestTrackerCooldown(unittest.TestCase):
    def setUp(self):
        # We need to mock the internal YOLO model created in __init__
        with patch('detector.YOLO') as mock_yolo:
            self.detector = PhoneDetector(model_instance=MagicMock(), pose_model_instance=MagicMock())

        self.detector.PERSON_COOLDOWN_SEC = 2 # Short cooldown for testing

    def test_cooldown_logic(self):
        # Mocking the save_evidence function to avoid thread spawning/files
        self.detector.save_evidence = MagicMock()

        current_time = time.time()

        # Scenario 1: First detection of Person 1 (Phone)
        track_id = 1
        should_save = True

        # Logic from detector.py
        if track_id != -1:
            if track_id not in self.detector.person_cooldowns:
                self.detector.person_cooldowns[track_id] = {}

            last_shot = self.detector.person_cooldowns[track_id].get('phone', 0)
            if current_time - last_shot < self.detector.PERSON_COOLDOWN_SEC:
                should_save = False

        if should_save:
            self.detector.person_cooldowns[track_id]['phone'] = current_time
            self.detector.save_evidence(None, 0,0,0,0, "TestCam", "PHONE")

        # Assertions
        self.assertTrue(should_save)
        self.detector.save_evidence.assert_called_once()
        self.detector.save_evidence.reset_mock()

        # Scenario 2: Immediate re-detection of Person 1
        should_save = True
        if track_id != -1:
            last_shot = self.detector.person_cooldowns[track_id].get('phone', 0)
            if current_time - last_shot < self.detector.PERSON_COOLDOWN_SEC:
                should_save = False

        self.assertFalse(should_save)
        self.detector.save_evidence.assert_not_called()

        # Scenario 3: Person 2 detection (New ID)
        track_id = 2
        should_save = True
        if track_id != -1:
            if track_id not in self.detector.person_cooldowns:
                self.detector.person_cooldowns[track_id] = {}
            last_shot = self.detector.person_cooldowns[track_id].get('phone', 0)
            if current_time - last_shot < self.detector.PERSON_COOLDOWN_SEC:
                should_save = False

        if should_save:
            self.detector.person_cooldowns[track_id]['phone'] = current_time
            self.detector.save_evidence(None, 0,0,0,0, "TestCam", "PHONE")

        self.assertTrue(should_save)
        self.detector.save_evidence.assert_called_once()

if __name__ == '__main__':
    unittest.main()
