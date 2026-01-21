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
             # Try to download if missing (optional helper)
             pass

        if not os.path.exists(mp_model_path):
             # Create dummy if running in benchmark without model
             # In production this raises error
             print(f"Warning: {mp_model_path} not found.")
             self.detector = None
        else:
            base_options = python.BaseOptions(model_asset_path=mp_model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)

        # Config
        self.EAR_THRESHOLD = 0.22
        self.SLEEP_TIME_THRESHOLD = 5.0
        self.MIN_FACE_SIZE = 48 # Pixels - Skip MediaPipe if crop is smaller than this
        
        self.HEAD_MOTION_THRESHOLD = 15.0
        self.MOTION_BUFFER_SIZE = 30

        self.state = {}

    def process_crop(self, crop, id_key="unknown", keypoints=None, crop_origin=(0,0)):
        if crop.size == 0:
            return "awake", {}

        current_time = time.time()

        if id_key not in self.state:
            self.state[id_key] = {
                'closed_start': None,
                'status': 'awake',
                'last_seen': current_time,
                'head_positions': [],
                'last_active_time': current_time,
                'ear_history': []
            }

        state = self.state[id_key]
        state['last_seen'] = current_time

        # --- STEP 1: EYES CHECK (MediaPipe) ---
        # Resolution Gate
        run_mediapipe = True
        if self.detector is None:
            run_mediapipe = False
        elif crop.shape[0] < self.MIN_FACE_SIZE or crop.shape[1] < self.MIN_FACE_SIZE:
            run_mediapipe = False
            
        if run_mediapipe:
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
            detection_result = self.detector.detect(mp_image)

            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                nose_landmark = landmarks[1]
                nose_position = (nose_landmark.x * crop.shape[1], nose_landmark.y * crop.shape[0])

                state['head_positions'].append(nose_position)
                if len(state['head_positions']) > self.MOTION_BUFFER_SIZE:
                    state['head_positions'].pop(0)

                left_ear = self._calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
                right_ear = self._calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])
                avg_ear = (left_ear + right_ear) / 2.0

                if 'ear_history' not in state: state['ear_history'] = []
                state['ear_history'].append(avg_ear)
                if len(state['ear_history']) > 300:
                    state['ear_history'].pop(0)

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
                        return "sleeping", {"ear": avg_ear, "duration": duration, "still": is_head_still, "source": "mediapipe"}
                    else:
                        return "drowsy", {"ear": avg_ear, "duration": duration, "still": is_head_still, "source": "mediapipe"}
                else:
                    state['closed_start'] = None
                    state['last_active_time'] = current_time
                    return "awake", {"ear": avg_ear, "still": is_head_still, "source": "mediapipe"}

        # --- STEP 2: ENHANCED POSTURE CHECK (YOLO) ---
        kpts = None
        if keypoints is not None:
            cx, cy = crop_origin
            kpts = keypoints.copy()
            kpts[:, 0] -= cx
            kpts[:, 1] -= cy
        else:
            # Fallback
            pose_results = self.pose_model.predict(crop, verbose=False, conf=0.5)
            if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                keypoints_data = pose_results[0].keypoints.xy.cpu().numpy()
                if len(keypoints_data) > 0:
                    kpts = keypoints_data[0]

        if kpts is not None:
            posture_result = self._check_sleep_posture(kpts, crop.shape)
            if posture_result['is_sleeping']:
                return "sleeping", {"reason": posture_result['reason'], "source": "yolo-pose", "details": posture_result}
            if posture_result['is_writing']:
                state['last_active_time'] = current_time
                return "awake", {"reason": "writing_detected", "source": "yolo-pose", "details": posture_result}

        return "awake", {"reason": "no_face_no_posture", "source": "fallback"}

    def _is_head_still(self, positions):
        if len(positions) < 10: return False
        positions_array = np.array(positions)
        x_std = np.std(positions_array[:, 0])
        y_std = np.std(positions_array[:, 1])
        return (x_std + y_std) < self.HEAD_MOTION_THRESHOLD

    def _check_sleep_posture(self, kpts, crop_shape):
        def has_pt(idx): return kpts[idx][0] > 0 and kpts[idx][1] > 0
        result = {'is_sleeping': False, 'is_writing': False, 'reason': None, 'details': {}}
        crop_height = crop_shape[0]
        
        has_shoulders = has_pt(5) and has_pt(6)
        has_face = has_pt(0) or has_pt(1) or has_pt(2) or has_pt(3) or has_pt(4)

        if has_shoulders and not has_face:
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            if shoulder_y < crop_height * 0.25:
                result['is_sleeping'] = True
                result['reason'] = "head_buried_high_shoulders"
                result['details']['shoulder_height_ratio'] = shoulder_y / crop_height
                return result
            else:
                return result
        
        if has_pt(0) and has_shoulders:
            nose_y = kpts[0][1]
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            if nose_y > shoulder_y + 30:
                result['is_sleeping'] = True
                result['reason'] = "slumped_forward"
                result['details']['slump_distance'] = nose_y - shoulder_y
                return result
        
        has_wrists = has_pt(9) or has_pt(10)
        if has_shoulders and has_wrists:
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            wrist_y = 0
            if has_pt(9) and has_pt(10): wrist_y = max(kpts[9][1], kpts[10][1])
            elif has_pt(9): wrist_y = kpts[9][1]
            else: wrist_y = kpts[10][1]
            
            if wrist_y > shoulder_y + 80:
                result['is_writing'] = True
                result['reason'] = "hands_on_desk"
                result['details']['hand_position'] = "active"
                if has_pt(0):
                    nose_y = kpts[0][1]
                    if nose_y > shoulder_y + 10 and nose_y < shoulder_y + 50: return result
        
        if has_pt(1) and has_pt(2):
            tilt = abs(kpts[1][1] - kpts[2][1])
            if tilt > 40:
                result['is_sleeping'] = True
                result['reason'] = "head_tilted"
                result['details']['tilt_amount'] = tilt
                return result
        
        if has_shoulders and (has_pt(11) or has_pt(12)):
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            hip_y = 0
            if has_pt(11) and has_pt(12): hip_y = (kpts[11][1] + kpts[12][1]) / 2
            elif has_pt(11): hip_y = kpts[11][1]
            else: hip_y = kpts[12][1]
            torso_length = abs(hip_y - shoulder_y)
            if torso_length < 60:
                result['is_sleeping'] = True
                result['reason'] = "collapsed_posture"
                result['details']['torso_length'] = torso_length
                return result
        
        if has_pt(0) and has_shoulders and (has_pt(7) or has_pt(8)):
            nose_y = kpts[0][1]
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            if nose_y > shoulder_y + 20 and nose_y < shoulder_y + 60:
                elbow_y = 0
                if has_pt(7) and has_pt(8): elbow_y = (kpts[7][1] + kpts[8][1]) / 2
                elif has_pt(7): elbow_y = kpts[7][1]
                else: elbow_y = kpts[8][1]
                if elbow_y > shoulder_y and elbow_y < shoulder_y + 100:
                    result['is_writing'] = True
                    result['reason'] = "reading_posture"
                    return result
        
        return result

    def _calculate_ear(self, landmarks, indices):
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
        if self.detector: self.detector.close()
