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
    def __init__(self, pose_model_path='yolo26n-pose.pt',
                 mp_model_path='models/face_landmarker.task',
                 camera_id=None):
        """
        Initialize Sleep Detector with private models and camera-specific state.

        Args:
            pose_model_path: Path to YOLO pose model (will be loaded as private instance)
            mp_model_path: Path to MediaPipe face landmarker model
            camera_id: Unique camera identifier for state isolation
        """
        self.camera_id = camera_id

        # 1. Load YOLO Pose (private instance for this camera)
        print(f"[Camera {camera_id}] Loading YOLO Pose model from {pose_model_path}...")
        try:
            self.pose_model = YOLO(pose_model_path)
            print(f"[Camera {camera_id}] YOLO Pose model loaded successfully.")
        except Exception as e:
            print(f"[Camera {camera_id}] Error loading YOLO Pose: {e}")
            self.pose_model = None

        # 2. Load MediaPipe Face Landmarker
        print(f"[Camera {camera_id}] Loading MediaPipe Face Landmarker from {mp_model_path}...")
        if not os.path.exists(mp_model_path):
            print(f"[Camera {camera_id}] Warning: {mp_model_path} not found. Face detection disabled.")
            self.detector = None
        else:
            try:
                base_options = python.BaseOptions(model_asset_path=mp_model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1
                )
                self.detector = vision.FaceLandmarker.create_from_options(options)
                print(f"[Camera {camera_id}] MediaPipe Face Landmarker loaded successfully.")
            except Exception as e:
                print(f"[Camera {camera_id}] Error loading MediaPipe: {e}")
                self.detector = None

        # Configuration
        self.EAR_THRESHOLD = 0.22
        self.SLEEP_TIME_THRESHOLD = 5.0
        self.MIN_FACE_SIZE = 48  # Pixels - Skip MediaPipe if crop is smaller
        
        self.HEAD_MOTION_THRESHOLD = 15.0
        self.MOTION_BUFFER_SIZE = 30

        # State tracking (camera-specific)
        self.state = {}

        print(f"[Camera {camera_id}] SleepDetector initialized.")

    def process_crop(self, crop, id_key="unknown", keypoints=None, crop_origin=(0,0)):
        """
        Process a person crop for sleep/drowsiness detection.

        Args:
            crop: Person crop image (numpy array)
            id_key: Person identifier (track ID)
            keypoints: Optional pre-computed pose keypoints (numpy array)
            crop_origin: (x, y) position of crop in original frame

        Returns:
            tuple: (status, metadata_dict)
                status: "awake", "drowsy", or "sleeping"
                metadata: dict with detection details
        """
        if crop.size == 0:
            return "awake", {}

        current_time = time.time()

        # Create unique state key per camera and person
        # Format: "cam<camera_id>_<id_key>"
        full_key = f"cam{self.camera_id}_{id_key}" if self.camera_id is not None else id_key

        if full_key not in self.state:
            self.state[full_key] = {
                'closed_start': None,
                'status': 'awake',
                'last_seen': current_time,
                'head_positions': [],
                'last_active_time': current_time,
                'ear_history': []
            }

        state = self.state[full_key]
        state['last_seen'] = current_time

        # Cleanup old state (not seen in 60 seconds)
        if len(self.state) > 100:
            cutoff = current_time - 60
            self.state = {k: v for k, v in self.state.items()
                         if v['last_seen'] > cutoff}

        # --- STEP 1: EYES CHECK (MediaPipe) ---
        run_mediapipe = True
        if self.detector is None:
            run_mediapipe = False
        elif crop.shape[0] < self.MIN_FACE_SIZE or crop.shape[1] < self.MIN_FACE_SIZE:
            run_mediapipe = False
            
        if run_mediapipe:
            try:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                detection_result = self.detector.detect(mp_image)

                if detection_result.face_landmarks:
                    landmarks = detection_result.face_landmarks[0]
                    nose_landmark = landmarks[1]
                    nose_position = (
                        nose_landmark.x * crop.shape[1],
                        nose_landmark.y * crop.shape[0]
                    )

                    state['head_positions'].append(nose_position)
                    if len(state['head_positions']) > self.MOTION_BUFFER_SIZE:
                        state['head_positions'].pop(0)

                    left_ear = self._calculate_ear(
                        landmarks, [33, 160, 158, 133, 153, 144]
                    )
                    right_ear = self._calculate_ear(
                        landmarks, [362, 385, 387, 263, 373, 380]
                    )
                    avg_ear = (left_ear + right_ear) / 2.0

                    if 'ear_history' not in state:
                        state['ear_history'] = []
                    state['ear_history'].append(avg_ear)
                    if len(state['ear_history']) > 300:
                        state['ear_history'].pop(0)

                    # Dynamic threshold based on person's baseline
                    current_threshold = self.EAR_THRESHOLD
                    if len(state['ear_history']) > 10:
                        baseline_ear = np.percentile(state['ear_history'], 90)
                        dynamic = baseline_ear * 0.8
                        current_threshold = min(self.EAR_THRESHOLD, max(0.12, dynamic))

                    is_head_still = self._is_head_still(state['head_positions'])

                    if avg_ear < current_threshold:
                        if state['closed_start'] is None:
                            state['closed_start'] = current_time

                        duration = current_time - state['closed_start']
                        if duration > self.SLEEP_TIME_THRESHOLD:
                            return "sleeping", {
                                "ear": avg_ear,
                                "threshold": current_threshold,
                                "duration": duration,
                                "still": is_head_still,
                                "source": "mediapipe"
                            }
                        else:
                            return "drowsy", {
                                "ear": avg_ear,
                                "threshold": current_threshold,
                                "duration": duration,
                                "still": is_head_still,
                                "source": "mediapipe"
                            }
                    else:
                        state['closed_start'] = None
                        state['last_active_time'] = current_time
                        return "awake", {
                            "ear": avg_ear,
                            "threshold": current_threshold,
                            "still": is_head_still,
                            "source": "mediapipe"
                        }

            except Exception as e:
                print(f"[Camera {self.camera_id}] MediaPipe processing error: {e}")
                # Fall through to posture check

        # --- STEP 2: ENHANCED POSTURE CHECK (YOLO) ---
        kpts = None
        if keypoints is not None:
            # Use pre-computed keypoints (more efficient)
            cx, cy = crop_origin
            kpts = keypoints.copy()
            kpts[:, 0] -= cx
            kpts[:, 1] -= cy
        else:
            # Fallback: Run pose inference on crop
            if self.pose_model is not None:
                try:
                    pose_results = self.pose_model.predict(crop, verbose=False, conf=0.5)
                    if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                        keypoints_data = pose_results[0].keypoints.xy.cpu().numpy()
                        if len(keypoints_data) > 0:
                            kpts = keypoints_data[0]
                except Exception as e:
                    print(f"[Camera {self.camera_id}] Pose inference error: {e}")

        if kpts is not None:
            posture_result = self._check_sleep_posture(kpts, crop.shape)
            if posture_result['is_sleeping']:
                return "sleeping", {
                    "reason": posture_result['reason'],
                    "source": "yolo-pose",
                    "details": posture_result
                }
            if posture_result['is_writing']:
                state['last_active_time'] = current_time
                return "awake", {
                    "reason": "writing_detected",
                    "source": "yolo-pose",
                    "details": posture_result
                }

        return "awake", {"reason": "no_face_no_posture", "source": "fallback"}

    def _is_head_still(self, positions):
        """
        Check if head is still based on position variance.

        Args:
            positions: List of (x, y) nose positions

        Returns:
            bool: True if head is still (low variance)
        """
        if len(positions) < 10:
            return False
        positions_array = np.array(positions)
        x_std = np.std(positions_array[:, 0])
        y_std = np.std(positions_array[:, 1])
        return (x_std + y_std) < self.HEAD_MOTION_THRESHOLD

    def _check_sleep_posture(self, kpts, crop_shape):
        """
        Check for sleep posture using YOLO keypoints.

        Detects patterns like:
        - Head buried (shoulders visible, face not)
        - Slumped forward
        - Head tilted significantly
        - Collapsed posture

        Also detects active postures like writing/reading.

        Args:
            kpts: YOLO pose keypoints (numpy array of shape [17, 2])
            crop_shape: Shape of the crop image (height, width, channels)

        Returns:
            dict: {is_sleeping, is_writing, reason, details}
        """
        def has_pt(idx):
            return kpts[idx][0] > 0 and kpts[idx][1] > 0

        result = {
            'is_sleeping': False,
            'is_writing': False,
            'reason': None,
            'details': {}
        }
        crop_height = crop_shape[0]
        
        has_shoulders = has_pt(5) and has_pt(6)
        has_face = has_pt(0) or has_pt(1) or has_pt(2) or has_pt(3) or has_pt(4)

        # Pattern 1: Head buried (shoulders high, no face)
        if has_shoulders and not has_face:
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            if shoulder_y < crop_height * 0.25:
                result['is_sleeping'] = True
                result['reason'] = "head_buried_high_shoulders"
                result['details']['shoulder_height_ratio'] = shoulder_y / crop_height
                return result
            else:
                return result
        
        # Pattern 2: Slumped forward (nose below shoulders)
        if has_pt(0) and has_shoulders:
            nose_y = kpts[0][1]
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            if nose_y > shoulder_y + 30:
                result['is_sleeping'] = True
                result['reason'] = "slumped_forward"
                result['details']['slump_distance'] = nose_y - shoulder_y
                return result
        
        # Pattern 3: Active desk work detection (hands on desk)
        has_wrists = has_pt(9) or has_pt(10)
        if has_shoulders and has_wrists:
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            wrist_y = 0
            if has_pt(9) and has_pt(10):
                wrist_y = max(kpts[9][1], kpts[10][1])
            elif has_pt(9):
                wrist_y = kpts[9][1]
            else:
                wrist_y = kpts[10][1]
            
            # Wrists significantly below shoulders = hands on desk
            if wrist_y > shoulder_y + 80:
                result['is_writing'] = True
                result['reason'] = "hands_on_desk"
                result['details']['hand_position'] = "active"

                # Additional check: head slightly forward (reading/writing posture)
                if has_pt(0):
                    nose_y = kpts[0][1]
                    if nose_y > shoulder_y + 10 and nose_y < shoulder_y + 50:
                        return result
        
        # Pattern 4: Head tilted significantly
        if has_pt(1) and has_pt(2):
            tilt = abs(kpts[1][1] - kpts[2][1])
            if tilt > 40:
                result['is_sleeping'] = True
                result['reason'] = "head_tilted"
                result['details']['tilt_amount'] = tilt
                return result
        
        # Pattern 5: Collapsed posture (short torso)
        if has_shoulders and (has_pt(11) or has_pt(12)):
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            hip_y = 0
            if has_pt(11) and has_pt(12):
                hip_y = (kpts[11][1] + kpts[12][1]) / 2
            elif has_pt(11):
                hip_y = kpts[11][1]
            else:
                hip_y = kpts[12][1]

            torso_length = abs(hip_y - shoulder_y)
            if torso_length < 60:
                result['is_sleeping'] = True
                result['reason'] = "collapsed_posture"
                result['details']['torso_length'] = torso_length
                return result
        
        # Pattern 6: Reading posture (head forward, elbows on desk)
        if has_pt(0) and has_shoulders and (has_pt(7) or has_pt(8)):
            nose_y = kpts[0][1]
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2

            # Head slightly forward
            if nose_y > shoulder_y + 20 and nose_y < shoulder_y + 60:
                elbow_y = 0
                if has_pt(7) and has_pt(8):
                    elbow_y = (kpts[7][1] + kpts[8][1]) / 2
                elif has_pt(7):
                    elbow_y = kpts[7][1]
                else:
                    elbow_y = kpts[8][1]

                # Elbows near shoulder level (leaning on desk)
                if elbow_y > shoulder_y and elbow_y < shoulder_y + 100:
                    result['is_writing'] = True
                    result['reason'] = "reading_posture"
                    return result
        
        return result

    def _calculate_ear(self, landmarks, indices):
        """
        Calculate Eye Aspect Ratio (EAR) for drowsiness detection.

        EAR = (vertical_1 + vertical_2) / (2 * horizontal)

        Lower values indicate closed eyes.

        Args:
            landmarks: MediaPipe face landmarks
            indices: List of 6 landmark indices for eye [outer, top1, top2, inner, bottom1, bottom2]

        Returns:
            float: Eye Aspect Ratio (typically 0.15-0.35)
        """
        def dist(i1, i2):
            p1 = landmarks[i1]
            p2 = landmarks[i2]
            return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

        vertical_1 = dist(indices[1], indices[5])
        vertical_2 = dist(indices[2], indices[4])
        horizontal = dist(indices[0], indices[3])

        if horizontal == 0:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def close(self):
        """Clean up resources."""
        if self.detector:
            self.detector.close()
