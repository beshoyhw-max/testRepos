import time
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import os

class SleepDetector:
    def __init__(self, pose_model_path='yolo11n-pose.pt', mp_model_path='models/face_landmarker.task', pose_model_instance=None):
        # 1. Load YOLO Pose
        if pose_model_instance:
            self.pose_model = pose_model_instance
        else:
            print("Loading YOLO Pose...")
            self.pose_model = YOLO(pose_model_path)

        # 2. Load MediaPipe Face Landmarker
        print("Loading MediaPipe Face Landmarker...")
        if not os.path.exists(mp_model_path):
             raise FileNotFoundError(f"MediaPipe model not found at {mp_model_path}")

        base_options = python.BaseOptions(model_asset_path=mp_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Config
        self.EAR_THRESHOLD = 0.22  # Below this is closed
        self.SLEEP_TIME_THRESHOLD = 2.0 # Seconds
        self.BLINK_THRESHOLD = 0.1 # Seconds

        self.state = {}

    def process_crop(self, crop, id_key="unknown"):
        """
        Analyzes a person crop for sleep signs.
        Refactored Order: MediaPipe (Eyes) -> If Failed -> YOLO (Posture)
        """
        if crop.size == 0:
            return "awake", {}

        current_time = time.time()

        # Initialize State for this ID
        if id_key not in self.state:
            self.state[id_key] = {
                'closed_start': None,
                'status': 'awake',
                'last_seen': current_time
            }

        state = self.state[id_key]
        state['last_seen'] = current_time

        # --- STEP 1: EYES CHECK (MediaPipe) ---
        # Convert to MP Image
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)

        detection_result = self.detector.detect(mp_image)

        if detection_result.face_landmarks:
            # Face Found! -> Check Eyes
            landmarks = detection_result.face_landmarks[0]

            # Calculate EAR
            left_ear = self._calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
            right_ear = self._calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])
            avg_ear = (left_ear + right_ear) / 2.0

            # Logic
            if avg_ear < self.EAR_THRESHOLD:
                # EYES CLOSED
                if state['closed_start'] is None:
                    state['closed_start'] = current_time

                duration = current_time - state['closed_start']
                if duration > self.SLEEP_TIME_THRESHOLD:
                    return "sleeping", {"ear": avg_ear, "duration": duration, "source": "mediapipe"}
                else:
                    return "drowsy", {"ear": avg_ear, "duration": duration, "source": "mediapipe"}
            else:
                # EYES OPEN -> Awake
                state['closed_start'] = None
                return "awake", {"ear": avg_ear, "source": "mediapipe"}

        # --- STEP 2: FALLBACK POSTURE CHECK (YOLO) ---
        # If we are here, MediaPipe failed to find a face (Occlusion / Head Down)

        # Run inference on crop
        # verbose=False to reduce logs
        pose_results = self.pose_model.predict(crop, verbose=False, conf=0.5)

        is_posture_sleep = False

        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            # Get Keypoints (xy)
            kpts = pose_results[0].keypoints.xy.cpu().numpy()[0] # Shape (17, 2)

            # Check if we have enough confidence (coords not 0,0)
            def has_pt(idx): return kpts[idx][0] > 0 and kpts[idx][1] > 0

            # Heuristic: Head Down
            # If Shoulders detected but Nose/Eyes NOT detected -> Head likely down/buried
            has_shoulders = has_pt(5) and has_pt(6)
            has_face = has_pt(0) or has_pt(1) or has_pt(2)

            if has_shoulders and not has_face:
                is_posture_sleep = True

            # Heuristic: Nose below Shoulders (Slumped forward deep)
            if has_pt(0) and has_shoulders:
                nose_y = kpts[0][1]
                shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
                if nose_y > shoulder_y:
                    is_posture_sleep = True

        if is_posture_sleep:
            return "sleeping", {"reason": "posture", "source": "yolo-pose"}

        return "awake", {"reason": "no_face_no_posture", "source": "fallback"}

    def _calculate_ear(self, landmarks, indices):
        # Indices in MediaPipe mesh
        # P1, P2, P3, P4, P5, P6
        # EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)

        def dist(i1, i2):
            p1 = landmarks[i1]
            p2 = landmarks[i2]
            return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

        vertical_1 = dist(indices[1], indices[5])
        vertical_2 = dist(indices[2], indices[4])
        horizontal = dist(indices[0], indices[3])

        if horizontal == 0: return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def close(self):
        self.detector.close()
