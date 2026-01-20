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
    def __init__(self, pose_model_path='yolo26n-pose.pt', mp_model_path='models/face_landmarker.task', pose_model_instance=None):
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
        
        # === NEW: Head Movement Tracking ===
        self.HEAD_MOTION_THRESHOLD = 15.0   # Pixels - if movement < this, person is very still
        self.MOTION_BUFFER_SIZE = 30  # Track last 30 frames (~1 second at 30fps)

        self.state = {}

    def process_batch(self, crops, id_keys):
        """
        Batched version of process_crop.
        Optimizes YOLO Pose inference by running it in a single batch for all fallbacks.

        Args:
            crops: List of crop images (numpy arrays)
            id_keys: List of ID strings corresponding to each crop

        Returns:
            List of (status, info_dict) corresponding to inputs.
        """
        results = [None] * len(crops)
        pose_batch_indices = []
        pose_batch_crops = []

        # Phase 1: MediaPipe (Sequential, CPU)
        for i, crop in enumerate(crops):
            if crop.size == 0:
                results[i] = ("awake", {})
                continue
            
            id_key = id_keys[i]
            current_time = time.time()

            # Initialize State
            if id_key not in self.state:
                self.state[id_key] = {
                    'closed_start': None,
                    'status': 'awake',
                    'last_seen': current_time,
                    'head_positions': [],
                    'last_active_time': current_time
                }

            state = self.state[id_key]
            state['last_seen'] = current_time

            # Run MediaPipe
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
            detection_result = self.detector.detect(mp_image)

            if detection_result.face_landmarks:
                # Face Found
                landmarks = detection_result.face_landmarks[0]
                
                # Update Motion
                nose_landmark = landmarks[1]
                nose_position = (nose_landmark.x * crop.shape[1], nose_landmark.y * crop.shape[0])
                state['head_positions'].append(nose_position)
                if len(state['head_positions']) > self.MOTION_BUFFER_SIZE:
                    state['head_positions'].pop(0)

                # EAR Check
                left_ear = self._calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
                right_ear = self._calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])
                avg_ear = (left_ear + right_ear) / 2.0
                is_head_still = self._is_head_still(state['head_positions'])
                
                if avg_ear < self.EAR_THRESHOLD:
                    if state['closed_start'] is None:
                        state['closed_start'] = current_time
                    duration = current_time - state['closed_start']

                    if duration > self.SLEEP_TIME_THRESHOLD:
                        results[i] = ("sleeping", {"ear": avg_ear, "duration": duration, "still": is_head_still, "source": "mediapipe"})
                    else:
                        results[i] = ("drowsy", {"ear": avg_ear, "duration": duration, "still": is_head_still, "source": "mediapipe"})
                else:
                    state['closed_start'] = None
                    state['last_active_time'] = current_time
                    results[i] = ("awake", {"ear": avg_ear, "still": is_head_still, "source": "mediapipe"})
            else:
                # MediaPipe Failed -> Add to Pose Batch
                pose_batch_indices.append(i)
                pose_batch_crops.append(crop)

        # Phase 2: YOLO Pose (Batched, GPU/CPU)
        if pose_batch_crops:
            # Run inference on all crops at once
            batch_pose_results = self.pose_model.predict(pose_batch_crops, verbose=False, conf=0.5)

            for j, pose_result in enumerate(batch_pose_results):
                orig_idx = pose_batch_indices[j]
                crop_shape = pose_batch_crops[j].shape

                # Logic copied from _process_crop_posture logic
                posture_result = None
                status = "awake"
                info = {"reason": "no_face_no_posture", "source": "fallback"}

                if pose_result.keypoints is not None:
                     keypoints_data = pose_result.keypoints.xy.cpu().numpy()
                     if len(keypoints_data) > 0:
                        kpts = keypoints_data[0]
                        posture_result = self._check_sleep_posture(kpts, crop_shape)

                        id_key = id_keys[orig_idx]
                        current_time = time.time()
                        state = self.state[id_key]

                        if posture_result['is_sleeping']:
                            status = "sleeping"
                            info = {
                                "reason": posture_result['reason'],
                                "source": "yolo-pose",
                                "details": posture_result
                            }
                        elif posture_result['is_writing']:
                            state['last_active_time'] = current_time
                            status = "awake"
                            info = {
                                "reason": "writing_detected",
                                "source": "yolo-pose",
                                "details": posture_result
                            }

                results[orig_idx] = (status, info)

        return results

    def process_crop(self, crop, id_key="unknown"):
        """
        Legacy wrapper for single crop processing.
        """
        results = self.process_batch([crop], [id_key])
        return results[0]

    def _is_head_still(self, positions):
        """
        Analyze head movement to distinguish sleeping from active reading/writing.
        
        Returns:
            bool: True if head is very still (likely sleeping)
        """
        if len(positions) < 10:
            return False  # Not enough data, assume active
        
        # Calculate movement variance
        positions_array = np.array(positions)
        
        # Standard deviation of positions (how much head moves)
        x_std = np.std(positions_array[:, 0])
        y_std = np.std(positions_array[:, 1])
        
        total_movement = x_std + y_std
        
        # If movement is very small, head is still (sleeping)
        return total_movement < self.HEAD_MOTION_THRESHOLD

    def _check_sleep_posture(self, kpts, crop_shape):
        """
        Enhanced posture analysis with writing detection.
        
        Args:
            kpts: YOLO pose keypoints (17, 2)
            crop_shape: Shape of person crop
            
        Returns:
            dict with 'is_sleeping', 'is_writing', 'reason'
        """
        def has_pt(idx): 
            return kpts[idx][0] > 0 and kpts[idx][1] > 0
        
        result = {
            'is_sleeping': False,
            'is_writing': False,
            'reason': None,
            'details': {}
        }
        
        # Keypoint indices (YOLO Pose COCO format)
        # 0: nose, 1-2: eyes, 3-4: ears
        # 5-6: shoulders, 7-8: elbows, 9-10: wrists
        # 11-12: hips
        
        crop_height = crop_shape[0]
        crop_width = crop_shape[1]
        
        # === 1. HEAD BURIED / DOWN (Your original - KEEP) ===
        has_shoulders = has_pt(5) and has_pt(6)
        has_face = has_pt(0) or has_pt(1) or has_pt(2)
        
        if has_shoulders and not has_face:
            result['is_sleeping'] = True
            result['reason'] = "head_buried"
            return result
        
        # === 2. DEEP SLUMP (Your original - ENHANCED) ===
        if has_pt(0) and has_shoulders:
            nose_y = kpts[0][1]
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            
            # Head significantly below shoulders
            if nose_y > shoulder_y + 30:
                result['is_sleeping'] = True
                result['reason'] = "slumped_forward"
                result['details']['slump_distance'] = nose_y - shoulder_y
                return result
        
        # === 3. NEW: WRITING/READING DETECTION ===
        # Check if hands are active on desk level (writing/typing)
        has_wrists = has_pt(9) or has_pt(10)
        
        if has_shoulders and has_wrists:
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            
            # Get lowest wrist position
            wrist_y = 0
            if has_pt(9) and has_pt(10):
                wrist_y = max(kpts[9][1], kpts[10][1])
            elif has_pt(9):
                wrist_y = kpts[9][1]
            else:
                wrist_y = kpts[10][1]
            
            # If wrists are significantly below shoulders = hands on desk
            # This indicates writing, typing, or reading with hands down
            if wrist_y > shoulder_y + 80:
                result['is_writing'] = True
                result['reason'] = "hands_on_desk"
                result['details']['hand_position'] = "active"
                
                # Additional check: if nose is also down but hands active = reading/writing
                if has_pt(0):
                    nose_y = kpts[0][1]
                    # Head slightly down + hands active = taking notes, NOT sleeping
                    if nose_y > shoulder_y + 10 and nose_y < shoulder_y + 50:
                        return result
        
        # === 4. NEW: HEAD TILT (Resting on desk) ===
        if has_pt(1) and has_pt(2):  # Both eyes visible
            left_eye_y = kpts[1][1]
            right_eye_y = kpts[2][1]
            tilt = abs(left_eye_y - right_eye_y)
            
            # Significant head tilt (> 40 pixels) = head resting sideways
            if tilt > 40:
                result['is_sleeping'] = True
                result['reason'] = "head_tilted"
                result['details']['tilt_amount'] = tilt
                return result
        
        # === 5. NEW: COLLAPSED POSTURE ===
        # Torso completely collapsed (hips very close to shoulders)
        if has_shoulders and (has_pt(11) or has_pt(12)):
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            
            hip_y = 0
            if has_pt(11) and has_pt(12):
                hip_y = (kpts[11][1] + kpts[12][1]) / 2
            elif has_pt(11):
                hip_y = kpts[11][1]
            else:
                hip_y = kpts[12][1]
            
            # Torso should be elongated; if collapsed = sleeping
            torso_length = abs(hip_y - shoulder_y)
            
            if torso_length < 60:  # Very short torso = collapsed
                result['is_sleeping'] = True
                result['reason'] = "collapsed_posture"
                result['details']['torso_length'] = torso_length
                return result
        
        # === 6. NEW: HEAD DOWN BUT STABLE (Reading position check) ===
        # If head is down but elbows are visible and positioned = reading stance
        if has_pt(0) and has_shoulders and (has_pt(7) or has_pt(8)):
            nose_y = kpts[0][1]
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            
            # Head moderately down
            if nose_y > shoulder_y + 20 and nose_y < shoulder_y + 60:
                # Check if elbows are at good reading height
                elbow_y = 0
                if has_pt(7) and has_pt(8):
                    elbow_y = (kpts[7][1] + kpts[8][1]) / 2
                elif has_pt(7):
                    elbow_y = kpts[7][1]
                else:
                    elbow_y = kpts[8][1]
                
                # Elbows between shoulders and hips = reading posture
                if elbow_y > shoulder_y and elbow_y < shoulder_y + 100:
                    result['is_writing'] = True
                    result['reason'] = "reading_posture"
                    return result
        
        return result

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