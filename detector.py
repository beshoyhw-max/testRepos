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
    def __init__(self, model_path='yolo26n.pt', pose_model_path='yolo26n-pose.pt',
                 output_dir="detections", cooldown_seconds=120,
                 consistency_threshold=3, camera_id=None):
        """
        Initialize Phone Detector with private models for thread safety.

        FIXED: No more shared models - each camera gets private instances.
        FIXED: Scale-invariant phone association with Hungarian algorithm.
        """
        
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # FIXED: Load private models (no sharing, no locks needed)
        print(f"[Camera {camera_id}] Loading private YOLO models...")
        self.model = YOLO(model_path)

        # FIXED: Pass camera_id for state isolation
        self.sleep_detector = SleepDetector(
            pose_model_path=pose_model_path,
            camera_id=camera_id
        )

        self.camera_id = camera_id
        self.PHONE_CLASS_ID = 67
        self.PERSON_CLASS_ID = 0
        self.COOLDOWN_SECONDS = cooldown_seconds
        self.CONSISTENCY_THRESHOLD = consistency_threshold
        
        self.cooldowns = {}
        self.streaks = {}
        self.last_display_data = []

        # FIXED: Track active IDs for better cleanup
        self.active_track_ids = set()

    def process_frame(self, frame, frame_count, skip_frames=5, save_screenshots=True,
                     conf_threshold=0.25, camera_name="Unknown"):
        """
        Process frame with optimized inference and scale-invariant association.
        """
        current_time = time.time()
        
        # FIXED: More aggressive cleanup (every 100 frames instead of 1000)
        if frame_count % 100 == 0:
            # Clean old cooldowns
            cutoff_time = current_time - (self.COOLDOWN_SECONDS * 2)
            self.cooldowns = {k: v for k, v in self.cooldowns.items() if v > cutoff_time}

            # Clean streaks for inactive IDs only
            self.streaks = {k: v for k, v in self.streaks.items()
                          if k[0] in self.active_track_ids}

            # Reset active IDs periodically
            if frame_count % 1000 == 0:
                self.active_track_ids.clear()

        global_status = "safe"
        screenshot_saved_global = False

        # --- INFERENCE STEP ---
        if frame_count % skip_frames == 0:
            self.last_display_data = []
            
            # 1. Vectorized Tracking (Person + Phone)
            classes_to_track = [self.PERSON_CLASS_ID, self.PHONE_CLASS_ID]
            
            try:
                # FIXED: No lock needed - private model
                results = self.model.track(
                    frame,
                    classes=classes_to_track,
                    conf=conf_threshold,
                    persist=True,
                    verbose=False,
                    imgsz=1280
                )
            except Exception as e:
                print(f"[{camera_name}] YOLO tracking error: {e}")
                results = []
            
            person_boxes = []
            phone_boxes = []

            if len(results) > 0 and results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    coords = box.xyxy[0].cpu().numpy()

                    if cls_id == self.PERSON_CLASS_ID:
                        if box.id is not None:
                            track_id = int(box.id.item())
                            person_boxes.append((*coords, track_id))
                            self.active_track_ids.add(track_id)
                    elif cls_id == self.PHONE_CLASS_ID:
                        conf = float(box.conf[0].item())
                        phone_boxes.append((*coords, conf))

            # 2. Vectorized Pose Estimation
            pose_keypoints_map = {}
            if hasattr(self.sleep_detector, 'pose_model'):
                try:
                    # FIXED: Private pose model, no race conditions
                    pose_results = self.sleep_detector.pose_model(frame, verbose=False, conf=0.5)
                    
                    if len(pose_results) > 0 and pose_results[0].boxes:
                        pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                        pose_kpts = pose_results[0].keypoints.xy.cpu().numpy()

                        pose_keypoints_map = self._associate_pose_to_persons(
                            person_boxes, pose_boxes, pose_kpts
                        )
                except Exception as e:
                    print(f"[{camera_name}] Pose inference error: {e}")

            # 3. FIXED: Scale-invariant phone association
            phone_map = self._associate_phones_to_persons(
                person_boxes, phone_boxes, pose_keypoints_map
            )

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
                        sleep_key = f"id_{track_id}"  # Camera ID handled internally
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
                if status == "texting":
                    violation_type = "texting"
                elif status == "sleeping":
                    violation_type = "sleeping"

                current_streak_val = 0

                if violation_type:
                    key = (track_id, violation_type)
                    self.streaks[key] = self.streaks.get(key, 0) + 1
                    current_streak_val = self.streaks[key]

                # Reset other violation types
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
                            self.save_evidence(
                                frame, x1, y1, x2, y2, camera_name,
                                type_str, track_id=track_id
                            )
                            screenshot_saved_global = True
                            self.cooldowns[key] = current_time
                            label += " [SAVED]"
                    else:
                        label += f" [{current_streak_val}/{self.CONSISTENCY_THRESHOLD}]"

                self.last_display_data.append((x1, y1, x2, y2, color, status, label))

        # --- DRAWING ---
        for (x1, y1, x2, y2, color, status, label) in self.last_display_data:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if status != "safe":
                status_text = "PHONE" if status == "texting" else status.upper()
                cv2.putText(frame, status_text, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if status == "texting":
                global_status = "texting"
            elif status == "sleeping" and global_status != "texting":
                global_status = "sleeping"

        return frame, global_status, screenshot_saved_global

    def _associate_phones_to_persons(self, person_boxes, phone_boxes, pose_keypoints_map):
        """
        FIXED: Scale-invariant phone association using Hungarian Algorithm.

        Maps phones to persons using:
        1. Normalized wrist distance (by person height)
        2. Normalized chest distance (fallback)
        3. Adaptive padding based on person size
        4. Overhead phone penalty (10000x cost)
        5. Hungarian algorithm for optimal global matching

        Returns:
            Dict mapping person_id to bool (has phone or not)
        """
        mapping = {}
        if not phone_boxes or not person_boxes:
            return mapping

        # Create Cost Matrix: Rows=Persons, Cols=Phones
        cost_matrix = []

        for person_bbox in person_boxes:
            p_x1, p_y1, p_x2, p_y2, p_id = person_bbox

            # FIXED: Calculate person dimensions for scale normalization
            person_width = p_x2 - p_x1
            person_height = p_y2 - p_y1
            person_area = person_width * person_height

            p_cx = (p_x1 + p_x2) / 2
            p_chest_y = p_y1 + person_height * 0.4  # 40% down from top

            row_costs = []

            for phone_bbox in phone_boxes:
                ph_x1, ph_y1, ph_x2, ph_y2, confidence = phone_bbox
                ph_cx = (ph_x1 + ph_x2) / 2
                ph_cy = (ph_y1 + ph_y2) / 2
                phone_area = (ph_x2 - ph_x1) * (ph_y2 - ph_y1)

                # --- COST CALCULATION ---
                cost = float('inf')

                # FIXED: Adaptive padding based on person size (not fixed 150px!)
                pad_x = person_width * 0.3   # 30% of width
                pad_y = person_height * 0.25  # 25% of height

                # Containment/Proximity Check with adaptive padding
                if (p_x1 - pad_x <= ph_cx <= p_x2 + pad_x and
                    p_y1 - pad_y <= ph_cy <= p_y2 + pad_y):

                    # Get keypoints for this person
                    kpts = pose_keypoints_map.get(p_id)
                    raw_distance = float('inf')

                    # Strategy A: Wrist Distance (most accurate)
                    if kpts is not None:
                        wrists = []
                        if kpts[9][0] > 0:   # Left wrist
                            wrists.append(kpts[9])
                        if kpts[10][0] > 0:  # Right wrist
                            wrists.append(kpts[10])

                        if wrists:
                            # Find minimum distance to any wrist
                            for w_pt in wrists:
                                d = math.hypot(ph_cx - w_pt[0], ph_cy - w_pt[1])
                                if d < raw_distance:
                                    raw_distance = d

                    # Strategy B: Chest Distance (fallback)
                    if raw_distance == float('inf'):
                        raw_distance = math.hypot(ph_cx - p_cx, ph_cy - p_chest_y)

                    # FIXED: Normalize distance by person height (scale-invariant!)
                    # This makes the cost comparable across different camera distances
                    normalized_cost = raw_distance / person_height
                    cost = normalized_cost

                    # --- CONTEXTUAL PENALTIES ---

                    # Penalty 1: Phone above person's head (wall mount, selfie)
                    # CRITICAL: This prevents false positives from background phones
                    if ph_cy < p_y1:
                        cost += 10000  # Massive penalty (keeps existing logic)

                    # Penalty 2: Phone significantly above eye level
                    # Even if not above head, if way above eyes = suspicious
                    if kpts is not None:
                        eyes = [kpts[1], kpts[2]]  # Left eye, right eye
                        eye_y = max((e[1] for e in eyes if e[1] > 0), default=0)

                        if eye_y > 0 and ph_cy < eye_y - person_height * 0.3:
                            # Phone 30% of height above eyes
                            cost *= 2.5  # Multiplicative penalty

                    # Penalty 3: Phone too small relative to person (likely false positive)
                    phone_to_person_ratio = phone_area / person_area
                    if phone_to_person_ratio < 0.005:  # < 0.5% of person area
                        cost *= 1.5

                    # Penalty 4: Phone outside shoulder width
                    if kpts is not None:
                        shoulders = []
                        if kpts[5][0] > 0:  # Left shoulder
                            shoulders.append(kpts[5])
                        if kpts[6][0] > 0:  # Right shoulder
                            shoulders.append(kpts[6])

                        if shoulders:
                            shoulder_x_min = min(s[0] for s in shoulders)
                            shoulder_x_max = max(s[0] for s in shoulders)
                            shoulder_width = shoulder_x_max - shoulder_x_min

                            # Phone significantly outside shoulder range
                            if (ph_cx < shoulder_x_min - shoulder_width * 0.3 or
                                ph_cx > shoulder_x_max + shoulder_width * 0.3):
                                cost *= 1.3

                row_costs.append(cost)
            cost_matrix.append(row_costs)

        # Hungarian Algorithm for optimal assignment
        C = np.array(cost_matrix)

        # Replace inf with large number for solver
        C[C == float('inf')] = 100000

        try:
            row_ind, col_ind = linear_sum_assignment(C)
        except Exception as e:
            print(f"Hungarian algorithm error: {e}")
            return mapping

        # FIXED: Adaptive threshold based on normalized cost
        # 0.5 = phone must be within 50% of person height
        # This automatically scales with camera distance!
        MAX_NORMALIZED_COST = 0.5  # Scale-invariant threshold

        for r, c in zip(row_ind, col_ind):
            cost_val = cost_matrix[r][c]

            # Check if assignment is valid (not infinity and below threshold)
            if cost_val < MAX_NORMALIZED_COST:
                p_id = person_boxes[r][4]  # Get person ID
                mapping[p_id] = True

        return mapping

    def _associate_pose_to_persons(self, person_boxes, pose_boxes, pose_kpts):
        """
        Associate pose detections with person detections using IoU + center fallback.
        """
        mapping = {}
        if len(person_boxes) == 0 or len(pose_boxes) == 0:
            return mapping
        
        for person_bbox in person_boxes:
            px1, py1, px2, py2, track_id = person_bbox
            p_area = (px2 - px1) * (py2 - py1)
            best_score = 0
            best_idx = -1

            for i, (pox1, poy1, pox2, poy2) in enumerate(pose_boxes):
                # Calculate IoU
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

                # Center Point Fallback (existing logic - keep it!)
                po_cx = (pox1 + pox2) / 2
                po_cy = (poy1 + poy2) / 2
                center_inside = (px1 <= po_cx <= px2) and (py1 <= po_cy <= py2)

                # Custom Scoring
                score = iou
                if center_inside:
                    if score < 0.4:
                        score = 0.4  # Boost score if center matches

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx != -1 and best_score > 0.3:
                mapping[track_id] = pose_kpts[best_idx]

        return mapping

    def save_evidence(self, frame, x1, y1, x2, y2, camera_name="Unknown",
                     detection_type="PHONE", track_id=None):
        """Save evidence screenshot with metadata overlay."""
        evidence_img = frame.copy()
        box_color = (0, 0, 255) if detection_type == "PHONE" else (255, 0, 0)
        cv2.rectangle(evidence_img, (x1, y1), (x2, y2), box_color, 3)
        
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(evidence_img, (0, 0), (evidence_img.shape[1], 40), (0,0,0), -1)
        header_text = f"{detection_type} | {camera_name} | {ts}"
        if track_id is not None:
            header_text += f" | ID: {track_id}"

        cv2.putText(evidence_img, header_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        timestamp_fn = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        safe_cam_name = "".join([c for c in camera_name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        id_str = f"_id{track_id}" if track_id is not None else ""
        filename = os.path.join(
            self.output_dir,
            f"evidence_{detection_type.lower()}_{safe_cam_name}_{timestamp_fn}{id_str}.jpg"
        )
        cv2.imwrite(filename, evidence_img)
        print(f"📸 EVIDENCE SAVED: {filename}")
