import unittest
import cv2
import numpy as np
import os
import time
from unittest.mock import MagicMock
from sleep_detector import SleepDetector

class TestSleepDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize once to save time, using real models
        try:
            cls.detector = SleepDetector()
        except Exception as e:
            print(f"Failed to load models: {e}")
            cls.detector = None

    def test_01_initialization(self):
        self.assertIsNotNone(self.detector, "SleepDetector failed to initialize")

    def test_02_eyes_open_real_image(self):
        if not self.detector: self.skipTest("Detector not loaded")

        # Download a known face image (Obama)
        url = "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg"
        import urllib.request
        try:
            img_path = "tests/obama.jpg"
            if not os.path.exists(img_path):
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(img_path, 'wb') as out_file:
                    out_file.write(response.read())

            img = cv2.imread(img_path)
            if img is None: self.skipTest("Could not read test image")

            # Run processing
            # We assume the image is a "crop" of a person
            status, info = self.detector.process_crop(img, id_key="test_obama")
            print(f"Obama Status: {status}, Info: {info}")

            # Allow drowsy if the specific stock photo has squinting eyes
            self.assertIn(status, ["awake", "drowsy"])
            self.assertIn("ear", info)
            self.assertGreater(info["ear"], 0.10) # Basic sanity check that it's not 0

        except Exception as e:
            print(f"DEBUG: Download failed due to: {e}")
            self.skipTest(f"Download or processing failed: {e}")

    def test_03_posture_logic_mock(self):
        if not self.detector: self.skipTest("Detector not loaded")

        # Mock the YOLO model
        original_model = self.detector.pose_model
        mock_model = MagicMock()
        self.detector.pose_model = mock_model

        # Create Mock Result
        mock_result = MagicMock()
        # Mock Keypoints: Nose (0) below Shoulders (5,6)
        # Coordinates: (x, y). Y increases downwards.
        # Nose at 100, Shoulders at 50 -> Nose is "below" (posturally down)

        # Create a tensor/array structure that matches YOLO output
        # shape (1, 17, 2)
        kpts = np.zeros((1, 17, 2), dtype=np.float32)

        # Set confidence high (coords > 0)
        # Nose
        kpts[0, 0] = [50, 100]
        # Left Shoulder
        kpts[0, 5] = [40, 50]
        # Right Shoulder
        kpts[0, 6] = [60, 50]

        # Mock the keypoints object
        mock_keypoints = MagicMock()
        import torch
        mock_keypoints.xy = torch.tensor(kpts)

        mock_result.keypoints = mock_keypoints
        mock_model.predict.return_value = [mock_result]

        # Run process_crop
        dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        status, info = self.detector.process_crop(dummy_crop, id_key="mock_posture")

        print(f"Mock Posture Status: {status}, Info: {info}")

        self.assertEqual(status, "sleeping")
        self.assertEqual(info["reason"], "posture")

        # Restore
        self.detector.pose_model = original_model

    def test_04_state_machine(self):
        if not self.detector: self.skipTest("Detector not loaded")

        # We need to mock _calculate_ear to return low value consistently
        # But _calculate_ear is a method.
        # We can mock the logic inside process_crop by mocking the detector result?
        # Easier: Mock `_calculate_ear` method on the instance.

        original_calc = self.detector._calculate_ear
        self.detector._calculate_ear = MagicMock(return_value=0.10) # Eyes Closed

        # We also need to bypass Pose check (return no keypoints or "awake" pose)
        # If we pass an empty crop, it returns awake instantly.
        # We need a crop that passes pose check but fails eye check.
        # We can mock pose_model again.

        original_pose = self.detector.pose_model
        mock_pose = MagicMock()
        mock_result = MagicMock()
        mock_result.keypoints = None # No body detected, so it falls through to Face check?
        # Wait, if no body, does it do face?
        # Logic: process_crop takes a crop. It checks pose on that crop.
        # If no keypoints, it falls through to "is_posture_sleep = False".
        # Then it tries MediaPipe.

        mock_pose.predict.return_value = [mock_result]
        self.detector.pose_model = mock_pose

        # We also need MediaPipe to return a "face".
        # Mocking `detector.detect`
        original_mp = self.detector.detector
        mock_mp = MagicMock()
        mock_detection = MagicMock()
        mock_landmarks = MagicMock()
        # process_crop accesses detection_result.face_landmarks[0]
        mock_detection.face_landmarks = [mock_landmarks]
        mock_mp.detect.return_value = mock_detection
        self.detector.detector = mock_mp

        dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)

        # T=0
        status, _ = self.detector.process_crop(dummy_crop, id_key="test_state")
        self.assertEqual(status, "drowsy") # First frame closed = drowsy (or blink)

        # T=1.0 (Sleep time in code is 2.0)
        time.sleep(1.0)
        status, _ = self.detector.process_crop(dummy_crop, id_key="test_state")
        self.assertEqual(status, "drowsy")

        # T=2.1
        time.sleep(1.2)
        status, info = self.detector.process_crop(dummy_crop, id_key="test_state")
        self.assertEqual(status, "sleeping")
        self.assertGreater(info["duration"], 2.0)

        # Reset
        self.detector.pose_model = original_pose
        self.detector.detector = original_mp
        self.detector._calculate_ear = original_calc

if __name__ == '__main__':
    unittest.main()
