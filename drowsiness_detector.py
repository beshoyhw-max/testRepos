import mediapipe as mp
import cv2
import numpy as np
import time

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class DrowsinessDetector:
    def __init__(self, model_path="face_landmarker.task"):
        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO, # Use VIDEO mode since we process frames sequentially
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True
        )
        self.landmarker = FaceLandmarker.create_from_options(self.options)

        # Landmark Indices (MediaPipe Face Mesh 468)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Thresholds
        self.EAR_THRESHOLD = 0.22 # Eyes Closed
        self.SLEEP_TIME_THRESHOLD = 2.0 # Seconds
        self.PITCH_THRESHOLD = 25 # Degrees (Head Down)
        self.HEAD_DOWN_TIME_THRESHOLD = 2.0 # Seconds

        # State Tracking: Map of Person ID -> State
        # Note: In VIDEO mode, MediaPipe tracks faces but IDs might not be persistent across restarts of tracking.
        # We rely on simple list index for now (assuming frame-to-frame consistency is handled by MP internal tracker).

        self.last_close_time = {} # FaceIndex -> timestamp
        self.last_head_down_time = {} # FaceIndex -> timestamp
        self.current_statuses = []

    def calculate_ear(self, landmarks, indices, w, h):
        coords = []
        for i in indices:
            lm = landmarks[i]
            coords.append(np.array([lm.x * w, lm.y * h]))

        p1, p2, p3, p4, p5, p6 = coords

        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        horiz = np.linalg.norm(p1 - p4)

        if horiz == 0: return 0.0
        ear = (v1 + v2) / (2.0 * horiz)
        return ear

    def get_head_pose(self, landmarks, w, h):
        # 3D Model Points (Generic Face)
        face_3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right Mouth corner
        ], dtype=np.float64)

        # Corresponding MediaPipe Landmarks
        # Nose: 1, Chin: 199, Eyes: 33, 263, Mouth: 61, 291
        idx_list = [1, 199, 33, 263, 61, 291]
        face_2d = []

        for idx in idx_list:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            face_2d.append([x, y])

        face_2d = np.array(face_2d, dtype=np.float64)

        # Camera Matrix (Approximate)
        focal_length = 1 * w
        cam_matrix = np.array([ [focal_length, 0, w/2],
                                [0, focal_length, h/2],
                                [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        if not success:
            return 0, 0

        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Q, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # angles[0] = Pitch, angles[1] = Yaw
        pitch = angles[0] * 360
        yaw = angles[1] * 360

        return pitch, yaw

    def process(self, frame, timestamp):
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect
        # Timestamp must be monotonically increasing in VIDEO mode
        ts_ms = int(timestamp * 1000)
        results = self.landmarker.detect_for_video(mp_image, ts_ms)

        self.current_statuses = []
        display_data = [] # (x, y, text, color)

        if results.face_landmarks:
            for face_id, landmarks in enumerate(results.face_landmarks):
                # Landmarks is a list of NormalizedLandmark objects

                # 1. EAR Calculation
                left_ear = self.calculate_ear(landmarks, self.LEFT_EYE, w, h)
                right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                # 2. Head Pose
                pitch, yaw = self.get_head_pose(landmarks, w, h)

                # Logic State
                status = "active"
                color = (0, 255, 0)

                # Check Sleep (Eyes Closed)
                if avg_ear < self.EAR_THRESHOLD:
                    if face_id not in self.last_close_time:
                        self.last_close_time[face_id] = timestamp

                    duration = timestamp - self.last_close_time[face_id]
                    if duration > self.SLEEP_TIME_THRESHOLD:
                        status = "sleeping"
                        color = (255, 0, 255) # Purple
                else:
                    self.last_close_time.pop(face_id, None)

                # Check Head Down (Pitch)
                if pitch > self.PITCH_THRESHOLD:
                    if face_id not in self.last_head_down_time:
                        self.last_head_down_time[face_id] = timestamp

                    duration = timestamp - self.last_head_down_time[face_id]
                    if duration > self.HEAD_DOWN_TIME_THRESHOLD:
                        status = "head_down"
                        color = (0, 165, 255) # Orange
                else:
                    self.last_head_down_time.pop(face_id, None)

                self.current_statuses.append(status)

                # Get Face Center for display (Nose Tip: 1)
                nx, ny = int(landmarks[1].x * w), int(landmarks[1].y * h)

                text = ""
                if status == "sleeping": text = "SLEEPING"
                if status == "head_down": text = "HEAD DOWN"

                if text:
                    display_data.append((nx, ny, text, color))

        return display_data, self.current_statuses
