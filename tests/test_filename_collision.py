import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import datetime
import numpy as np

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies
sys.modules['ultralytics'] = MagicMock()
sys.modules['sleep_detector'] = MagicMock()

from detector import PhoneDetector

class TestFilenameCollision(unittest.TestCase):
    def setUp(self):
        self.detector = PhoneDetector(
            output_dir="tests/output_collision",
            model_instance=MagicMock(),
            pose_model_instance=MagicMock()
        )
        # Mock cv2.imwrite
        self.mock_imwrite = MagicMock()
        sys.modules['cv2'].imwrite = self.mock_imwrite

        # Mock frame
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_filename_uniqueness_with_ids(self):
        """
        Demonstrate that providing track_id prevents collision even at same second.
        """
        # Freeze time
        fixed_time = datetime.datetime(2023, 10, 27, 10, 0, 0)

        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = fixed_time
            mock_datetime.side_effect = lambda *args, **kw: fixed_time

            # Call 1: Person A (ID 1)
            self.detector.save_evidence(self.frame, 0,0,10,10, "Cam1", "PHONE", track_id=1)
            args1, _ = self.mock_imwrite.call_args
            filename1 = args1[0]

            # Call 2: Person B (ID 2)
            self.detector.save_evidence(self.frame, 20,20,30,30, "Cam1", "PHONE", track_id=2)
            args2, _ = self.mock_imwrite.call_args
            filename2 = args2[0]

            print(f"File 1: {filename1}")
            print(f"File 2: {filename2}")

            # This assertion confirms the FIX
            self.assertNotEqual(filename1, filename2, "Filenames should be unique due to ID inclusion")
            self.assertIn("_id1.jpg", filename1)
            self.assertIn("_id2.jpg", filename2)

if __name__ == '__main__':
    unittest.main()
