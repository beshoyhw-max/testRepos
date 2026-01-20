from unittest.mock import MagicMock
import sys

# Mock cv2 and ultralytics
sys.modules['cv2'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['sleep_detector'] = MagicMock()

from detector import PhoneDetector

# Instantiate
d = PhoneDetector(model_instance=MagicMock(), pose_model_instance=MagicMock())

# Check attribute
if hasattr(d, 'person_cooldowns'):
    print("SUCCESS: person_cooldowns exists")
else:
    print("FAILURE: person_cooldowns missing")
