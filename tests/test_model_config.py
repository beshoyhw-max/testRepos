
import sys
import unittest
from unittest.mock import MagicMock

# Mock external dependencies before importing local modules
sys.modules['ultralytics'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.tasks'] = MagicMock()
sys.modules['mediapipe.tasks.python'] = MagicMock()
sys.modules['mediapipe.tasks.python.vision'] = MagicMock()

# Mock the YOLO class specifically to track calls
mock_yolo = MagicMock()
sys.modules['ultralytics'].YOLO = mock_yolo

# Now import the local modules
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from camera_manager import CameraManager
from detector import PhoneDetector
from sleep_detector import SleepDetector

class TestModelVersions(unittest.TestCase):
    def test_camera_manager_models(self):
        """Verify CameraManager uses YOLO26 models"""
        # Reset mock calls
        mock_yolo.reset_mock()

        # Instantiate CameraManager
        # Mock load_config_and_start to avoid side effects
        with unittest.mock.patch('camera_manager.CameraManager.load_config_and_start'):
            cm = CameraManager(config_file="dummy_cameras.json")

        calls = [str(call) for call in mock_yolo.mock_calls]

        # Verify specific models are requested
        found_detection = any("yolo26n.pt" in c for c in calls)
        found_pose = any("yolo26n-pose.pt" in c for c in calls)

        self.assertTrue(found_detection, f"CameraManager did not load yolo26n.pt. Calls: {calls}")
        self.assertTrue(found_pose, f"CameraManager did not load yolo26n-pose.pt. Calls: {calls}")

    def test_detector_defaults(self):
        """Verify PhoneDetector uses YOLO26 models by default"""
        mock_yolo.reset_mock()

        with unittest.mock.patch('os.path.exists', return_value=True):
             pd = PhoneDetector(model_instance=None, pose_model_instance=None)

        calls = [str(call) for call in mock_yolo.mock_calls]

        found_det = any("yolo26s.pt" in c for c in calls)
        self.assertTrue(found_det, f"PhoneDetector did not load yolo26s.pt. Calls: {calls}")

    def test_sleep_detector_defaults(self):
        """Verify SleepDetector uses YOLO26 models by default"""
        mock_yolo.reset_mock()

        with unittest.mock.patch('os.path.exists', return_value=True):
            sd = SleepDetector(pose_model_instance=None)

        calls = [str(call) for call in mock_yolo.mock_calls]

        found_pose = any("yolo26n-pose.pt" in c for c in calls)
        self.assertTrue(found_pose, f"SleepDetector did not load yolo26n-pose.pt. Calls: {calls}")

if __name__ == '__main__':
    unittest.main()
