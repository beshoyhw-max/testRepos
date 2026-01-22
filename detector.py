import cv2
import time
import os
import math
import datetime
from ultralytics import YOLO
import threading
import numpy as np
from scipy.optimize import linear_sum_assignment
from sleep_detector import SleepDetector

class PhoneDetector:
    def __init__(self, model_path='yolo26n.pt', output_dir="detections", cooldown_seconds=120, consistency_threshold=3, model_instance=None, pose_model_instance=None, lock=None):
        
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Handling Shared Model
        self.lock = lock
        if model_instance:
            self.model = model_instance
        else:
            self.model = YOLO(model_path)
            
        # Initialize Sleep Detector
        self.sleep_detector = SleepDetector(pose_model_instance=pose_model_instance)

        self.PHONE_CLASS_ID = 67
        self.PERSON_CLASS_ID = 0
        self.COOLDOWN_SECONDS = cooldown_seconds
        self.CONSISTENCY_THRESHOLD = consistency_threshold
        
        self.cooldowns = {}
        self.streaks = {}
        self.last_display_data = []

    def process_frame(self, frame, frame_count, skip_frames=5, save_screenshots=True, conf_threshold=0.25, camera_name="Unknown"):
        """
        Optimized Process Frame:
        - Vectorized inference for Persons AND Phones (Single Pass)
        - Vectorized inference for Pose (Single Pass)
        - Geometric association
        """
        current_time = time.time()
        
        # Cleanup
        if frame_count % 1000 == 0:
            self.cooldowns = {k: v for k, v in self.cooldowns.items() if (current_time - v) < self.COOLDOWN_SECONDS * 2}
            self.streaks = {}

        global_status = "safe"
        screenshot_saved_global = False

        # --- HEAVY INFERENCE STEP ---
        if frame_count % skip_frames == 0:
            self.last_display_data = []
            
            # 1. Vectorized Tracking (Person + Phone)
            classes_to_track = [self.PERSON_CLASS_ID, self.PHONE_CLASS_ID]
            
            if self.lock:
                with self.lock:
                    results = self.model.track(frame, classes=classes_to_track, conf=conf_threshold, persist=True, verbose=False, imgsz=1280)
            else:
                results = self.model.track(frame, classes=classes_to_track, conf=conf_threshold, persist=True, verbose=False, imgsz=1280)
            
            person_boxes = [] # (x1, y1, x2, y2, id)
            phone_boxes = []  # (x1, y1, x2, y2, conf)

            if len(results) > 0 and results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    coords = box.xyxy[0].cpu().numpy()

                    if cls_id == self.PERSON_CLASS_ID:
                        if box.id is not None:
                            track_id = int(box.id.item())
                            person_boxes.append((*coords, track_id))
                    elif cls_id == self.PHONE_CLASS_ID:
                        conf = float(box.conf[0].item())
                        phone_boxes.append((*coords, conf))

            # 2. Vectorized Pose Estimation (Full Frame)
            pose_keypoints_map = {}
            if hasattr(self.sleep_detector, 'pose_model'):
                try:
                    pose_results = self.sleep_detector.pose_model(frame, verbose=False, conf=0.5)
                    
                    if len(pose_results) > 0 and pose_results[0].boxes:
                        pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                        pose_kpts = pose_results[0].keypoints.xy.cpu().numpy()

                        # Match Pose to Persons
                        pose_keypoints_map = self._associate_pose_to_persons(person_boxes, pose_boxes, pose_kpts)
                except Exception as e:
                    print(f"Pose Inference Error: {e}")

            # 3. Map Phones to Persons (Using Keypoints if available)
            phone_map = self._associate_phones_to_persons(person_boxes, phone_boxes, pose_keypoints_map)

            # 4. Process Each Person
            for p_box in person_boxes:
                x1, y1, x2, y2, track_id = map(int, p_box)

                status = "safe"
                color = (0, 255, 0)
                label = f"ID: {track_id}"

                # Check Phone Map
                has_phone = phone_map.get(track_id, False)

                if has_phone:
                    status = "texting"
                    color = (0, 0, 255)
                    global_status = "texting"
                else:
                    # Sleep Detection
                    h, w, _ = frame.shape
                    pad = 20
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(w, x2 + pad)
                    cy2 = min(h, y2 + pad)
                    person_crop = frame[cy1:cy2, cx1:cx2]
                    
                    if person_crop.size > 0:
                        sleep_key = f"{camera_name}_id_{track_id}"
                        kpts = pose_keypoints_map.get(track_id)
                        
                        sleep_status, _ = self.sleep_detector.process_crop(
                            person_crop,
                            id_key=sleep_key,
                            keypoints=kpts,
                            crop_origin=(cx1, cy1)
                        )
                        
                        if sleep_status == "sleeping":
                            status = "sleeping"
                            color = (255, 0, 0)
                            if global_status != "texting":
                                global_status = "sleeping"
                        elif sleep_status == "drowsy":
                            color = (0, 255, 255)

                # --- VERIFICATION & SAVING ---
                violation_type = None
                if status == "texting": violation_type = "texting"
                elif status == "sleeping": violation_type = "sleeping"

                current_streak_val = 0

                if violation_type:
                    key = (track_id, violation_type)
                    self.streaks[key] = self.streaks.get(key, 0) + 1
                    current_streak_val = self.streaks[key]

                possible_types = ["texting", "sleeping"]
                for v_type in possible_types:
                    if v_type != violation_type:
                        self.streaks[(track_id, v_type)] = 0

                if save_screenshots and violation_type:
                    if current_streak_val >= self.CONSISTENCY_THRESHOLD:
                        key = (track_id, violation_type)
                        last_time = self.cooldowns.get(key, 0)

                        if (current_time - last_time) > self.COOLDOWN_SECONDS:
                            type_str = "PHONE" if violation_type == "texting" else "SLEEP"
                            self.save_evidence(frame, x1, y1, x2, y2, camera_name, type_str, track_id=track_id)
                            screenshot_saved_global = True
                            self.cooldowns[key] = current_time
                            label += " [SAVED]"
                    else:
                        label += f" [{current_streak_val}/{self.CONSISTENCY_THRESHOLD}]"

                self.last_display_data.append((x1, y1, x2, y2, color, status, label))

        # --- DRAWING ---
        for (x1, y1, x2, y2, color, status, label) in self.last_display_data:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if status != "safe":
                status_text = "PHONE" if status == "texting" else status.upper()
                cv2.putText(frame, status_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if status == "texting": global_status = "texting"
            elif status == "sleeping" and global_status != "texting": global_status = "sleeping"

        return frame, global_status, screenshot_saved_global

    def _associate_phones_to_persons(self, person_boxes, phone_boxes, pose_keypoints_map):
        """
        Maps phones to persons using Keypoints (High Accuracy) or Bounding Box (Fallback).
        Uses Hungarian Algorithm for global optimization.
        """
        mapping = {}
        if not phone_boxes or not person_boxes: return mapping

        # Create Cost Matrix: Rows=Persons, Cols=Phones
        cost_matrix = []

        for p in person_boxes:
            p_x1, p_y1, p_x2, p_y2, p_id = p
            p_cx = (p_x1 + p_x2) / 2
            # p_chest_y is used as fallback center
            p_chest_y = p_y1 + (p_y2 - p_y1) * 0.4

            row_costs = []
            for ph in phone_boxes:
                ph_x1, ph_y1, ph_x2, ph_y2, _ = ph
                ph_cx = (ph_x1 + ph_x2) / 2
                ph_cy = (ph_y1 + ph_y2) / 2

                # --- COST CALCULATION ---
                cost = float('inf')

                # 1. Containment / Proximity Check
                pad = 150 # Increased slightly for robustness
                if (p_x1 - pad <= ph_cx <= p_x2 + pad and p_y1 - pad <= ph_cy <= p_y2 + pad):

                    # Base Cost: Distance
                    kpts = pose_keypoints_map.get(p_id)
                    dist = float('inf')

                    # Strategy A: Wrist Distance
                    if kpts is not None:
                        wrists = []
                        if kpts[9][0] > 0: wrists.append(kpts[9])
                        if kpts[10][0] > 0: wrists.append(kpts[10])

                        if wrists:
                            for w_pt in wrists:
                                d = math.hypot(ph_cx - w_pt[0], ph_cy - w_pt[1])
                                if d < dist: dist = d

                    # Strategy B: Chest Distance (Fallback)
                    if dist == float('inf'):
                        dist = math.hypot(ph_cx - p_cx, ph_cy - p_chest_y)

                    cost = dist

                    # 2. Vertical Penalty (Phone above head)
                    # Requirement: "Add a massive distance penalty if phone.y < head.y"
                    # Using Bounding Box Top (p_y1) as Head Top approximation.
                    if ph_cy < p_y1:
                        cost += 10000 # Massive penalty

                row_costs.append(cost)
            cost_matrix.append(row_costs)

        # Hungarian Algorithm
        C = np.array(cost_matrix)

        # Replace inf with a very large number for the solver
        C[C == float('inf')] = 100000

        row_ind, col_ind = linear_sum_assignment(C)

        # Filter assignments
        MAX_DIST = 300 # Pixels

        for r, c in zip(row_ind, col_ind):
            cost_val = cost_matrix[r][c]

            if cost_val < MAX_DIST:
                # person_boxes[r] -> get ID
                p_id = person_boxes[r][4] # Index 4 is ID
                mapping[p_id] = True

        return mapping

    def _associate_pose_to_persons(self, person_boxes, pose_boxes, pose_kpts):
        mapping = {}
        if len(person_boxes) == 0 or len(pose_boxes) == 0:
            return mapping
        
        for p_box in person_boxes:
            px1, py1, px2, py2, track_id = p_box
            p_area = (px2 - px1) * (py2 - py1)
            best_score = 0
            best_idx = -1

            for i, (pox1, poy1, pox2, poy2) in enumerate(pose_boxes):
                ix1 = max(px1, pox1)
                iy1 = max(py1, poy1)
                ix2 = min(px2, pox2)
                iy2 = min(py2, poy2)

                iou = 0
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    po_area = (pox2 - pox1) * (poy2 - poy1)
                    union_area = p_area + po_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0

                # Center Point Fallback
                po_cx = (pox1 + pox2) / 2
                po_cy = (poy1 + poy2) / 2
                center_inside = (px1 <= po_cx <= px2) and (py1 <= po_cy <= py2)

                # Custom Scoring
                score = iou
                if center_inside:
                    # If IoU is low but center is inside, boost score to be acceptable
                    if score < 0.4:
                        score = 0.4

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx != -1 and best_score > 0.3:
                mapping[track_id] = pose_kpts[best_idx]
        return mapping

    def save_evidence(self, frame, x1, y1, x2, y2, camera_name="Unknown", detection_type="PHONE", track_id=None):
        evidence_img = frame.copy()
        box_color = (0, 0, 255) if detection_type == "PHONE" else (255, 0, 0)
        cv2.rectangle(evidence_img, (x1, y1), (x2, y2), box_color, 3)
        
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(evidence_img, (0, 0), (evidence_img.shape[1], 40), (0,0,0), -1)
        header_text = f"{detection_type} | {camera_name} | {ts}"
        if track_id is not None: header_text += f" | ID: {track_id}"

        cv2.putText(evidence_img, header_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        timestamp_fn = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        safe_cam_name = "".join([c for c in camera_name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        id_str = f"_id{track_id}" if track_id is not None else ""
        filename = os.path.join(self.output_dir, f"evidence_{detection_type.lower()}_{safe_cam_name}_{timestamp_fn}{id_str}.jpg")
        cv2.imwrite(filename, evidence_img)
        print(f"📸 EVIDENCE SAVED: {filename}")
