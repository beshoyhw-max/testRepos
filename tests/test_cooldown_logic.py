import unittest
from unittest.mock import MagicMock
import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies
sys.modules['ultralytics'] = MagicMock()
sys.modules['sleep_detector'] = MagicMock()

from detector import PhoneDetector

class TestCooldownLogic(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_pose_model = MagicMock()

        # Initialize detector
        self.detector = PhoneDetector(
            model_instance=self.mock_model,
            pose_model_instance=self.mock_pose_model,
            output_dir="tests/output",
            cooldown_seconds=2 # Short cooldown for testing
        )
        # Mock save_evidence to avoid file I/O
        self.detector.save_evidence = MagicMock()

        # Inject mock sleep detector
        self.detector.sleep_detector = MagicMock()

    def test_cooldowns(self):
        """
        Test that evidence is saved only after cooldown expires per (ID, Type).
        """
        # We need to monkey-patch the logic if we were testing existing code,
        # but here we are preparing the test for the NEW code.
        # This test assumes process_frame uses self.model.track and handles IDs.

        # MOCK SETUP
        # Helper to create mock results
        class MockBox:
            def __init__(self, id_val, cls_val):
                self.id = None if id_val is None else MagicMock()
                if self.id:
                    self.id.int.return_value = id_val
                    self.id.item.return_value = id_val
                self.cls = MagicMock()
                self.cls.item.return_value = cls_val
                self.xyxy = MagicMock()
                self.xyxy.__iter__.return_value = [100, 100, 200, 200]
                self.xyxy.cpu().numpy().return_value = np.array([[100, 100, 200, 200]]) # For iterating
                self.xyxy.__getitem__.return_value = [100, 100, 200, 200]

        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes
                self.__len__ = lambda s: len(boxes)

        # SCENARIO:
        # Frame 1: Person 1 -> Texting
        # Frame 2: Person 1 -> Texting (Should be blocked by cooldown)
        # Frame 3: Person 1 -> Sleeping (Should be allowed, different type)
        # ... Wait 2.1 seconds ...
        # Frame N: Person 1 -> Texting (Should be allowed)

        # --- Frame 1 ---
        # Mock Tracking: Person ID 1 found
        track_res = MockResult([MockBox(1, 0)])
        self.mock_model.track.return_value = [track_res]

        # Mock Phone Detection (crop predict): Phone found
        phone_res = MockResult([MockBox(None, 67)])
        self.mock_model.predict.return_value = [phone_res]

        # Mock Sleep: Awake
        self.detector.sleep_detector.process_crop.return_value = ("awake", {})

        # Run
        self.detector.process_frame(np.zeros((720,1280,3), np.uint8), 0, skip_frames=1)

        # Assertions
        # Should have saved TEXTING for ID 1
        self.detector.save_evidence.assert_called()
        self.assertEqual(self.detector.save_evidence.call_count, 1)
        args, _ = self.detector.save_evidence.call_args
        self.assertEqual(args[6], "PHONE") # detection_type arg

        # Reset mock
        self.detector.save_evidence.reset_mock()

        # --- Frame 2 (Immediate follow up) ---
        self.detector.process_frame(np.zeros((720,1280,3), np.uint8), 1, skip_frames=1)

        # Should NOT save (Cooldown)
        self.detector.save_evidence.assert_not_called()

        # --- Frame 3 (Switch to Sleeping) ---
        # Mock Phone: None
        self.mock_model.predict.return_value = [MockResult([])]
        # Mock Sleep: Sleeping
        self.detector.sleep_detector.process_crop.return_value = ("sleeping", {"reason": "test"})

        self.detector.process_frame(np.zeros((720,1280,3), np.uint8), 2, skip_frames=1)

        # Should save SLEEPING for ID 1 (Different type)
        self.detector.save_evidence.assert_called()
        args, _ = self.detector.save_evidence.call_args
        self.assertEqual(args[6], "SLEEP")

        # Reset
        self.detector.save_evidence.reset_mock()

        # --- Wait for Cooldown ---
        time.sleep(2.1)

        # --- Frame 4 (Texting again) ---
        # Mock Phone: Found
        self.mock_model.predict.return_value = [phone_res]
        # Mock Sleep: Awake
        self.detector.sleep_detector.process_crop.return_value = ("awake", {})

        self.detector.process_frame(np.zeros((720,1280,3), np.uint8), 3, skip_frames=1)

        # Should save TEXTING again
        self.detector.save_evidence.assert_called()


if __name__ == '__main__':
    unittest.main()
