import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need to patch YOLO and SleepDetector BEFORE importing detector
# because detector imports them at top level.
# Actually detector imports them, but uses them in __init__
from detector import PhoneDetector

class TestDetectorLogic(unittest.TestCase):
    def setUp(self):
        # Patch Ultralytics YOLO to avoid loading real models
        self.yolo_patcher = patch('detector.YOLO')
        self.mock_yolo_cls = self.yolo_patcher.start()
        self.mock_yolo_instance = MagicMock()
        self.mock_yolo_cls.return_value = self.mock_yolo_instance

        # Patch SleepDetector to avoid MediaPipe/Pose issues
        self.sleep_patcher = patch('detector.SleepDetector')
        self.mock_sleep_cls = self.sleep_patcher.start()
        self.mock_sleep_instance = MagicMock()
        self.mock_sleep_cls.return_value = self.mock_sleep_instance

        # Initialize detector with camera_id
        self.detector = PhoneDetector(camera_id=1)

    def tearDown(self):
        self.yolo_patcher.stop()
        self.sleep_patcher.stop()

    def test_associate_phones_to_persons_hungarian(self):
        """
        Test that the matching uses a global optimization (Hungarian Algorithm).
        User's version uses normalized cost, but the logic remains:
        Minimize total cost.

        Scenario:
        Person A (Small, far away): 100x200 box.
        Person B (Large, close): 200x400 box.
        """
        # Person boxes: [x1, y1, x2, y2, id]
        # Person 1: 100x200 (Area=20000). Center ~ (150, 200)
        p1 = [100, 100, 200, 300, 1]

        # Person 2: 200x400 (Area=80000). Center ~ (400, 400)
        p2 = [300, 200, 500, 600, 2]

        persons = [p1, p2]

        # Phone boxes: [x1, y1, x2, y2, conf]
        # Phone 1: Near Person 1 chest.
        # P1 Chest Y ~ 100 + 200*0.4 = 180. Center (150, 180).
        ph1 = [145, 175, 155, 185, 0.9] # Center (150, 180)

        # Phone 2: Near Person 2 chest.
        # P2 Chest Y ~ 200 + 400*0.4 = 360. Center (400, 360).
        ph2 = [395, 355, 405, 365, 0.9] # Center (400, 360)

        phones = [ph1, ph2]

        mapping = self.detector._associate_phones_to_persons(persons, phones, {})

        self.assertTrue(mapping.get(1))
        self.assertTrue(mapping.get(2))

    def test_vertical_penalty(self):
        """
        Test that a phone physically above the person's head (y < person_y1)
        is NOT assigned, due to massive penalty.
        """
        # Person 1: Box (100, 300, 200, 500). Head top is y=300.
        persons = [
            [100, 300, 200, 500, 1]
        ]

        # Phone: Center (150, 200).
        # This is above the person's head (200 < 300).
        phone_above = [140, 190, 160, 210, 0.9]

        phones = [phone_above]

        mapping = self.detector._associate_phones_to_persons(persons, phones, {})

        # Should NOT match because of penalty
        self.assertIsNone(mapping.get(1))

    def test_associate_pose_center_fallback(self):
        """
        Test matching pose to person using Center Point Fallback when IoU is low.
        """
        # Person Box: Large box
        persons = [
            [100, 100, 500, 500, 1]
        ]

        # Pose Box: Small box inside person box.
        pose_box = [275, 275, 325, 325]
        pose_kpts = [np.zeros((17, 2))] # Dummy keypoints

        pose_boxes = [pose_box]
        pose_kpts_list = [pose_kpts]

        mapping = self.detector._associate_pose_to_persons(persons, pose_boxes, pose_kpts_list)

        # Should match due to center point fallback
        self.assertIn(1, mapping)

if __name__ == '__main__':
    unittest.main()
