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
        Initialize Phone Detector with advanced temporal and biomechanical logic.

        Features:
        - Motion Vector Alignment (physics-based validation)
        - Bucket-Brigade Hysteresis (temporal stability)
        - Scene Memory (static object suppression)
        - Sticky Association (tracking continuity)
        - Elbow Angle Validation (biomechanical constraints)
        """
        
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print(f"[Camera {camera_id}] Loading private YOLO models...")
        self.model = YOLO(model_path)

        self.sleep_detector = SleepDetector(
            pose_model_path=pose_model_path,
            camera_id=camera_id
        )

        self.camera_id = camera_id
        self.PHONE_CLASS_ID = 67
        self.PERSON_CLASS_ID = 0
        self.COOLDOWN_SECONDS = cooldown_seconds

        # Bucket-Brigade Hysteresis Configuration
        self.HYSTERESIS_MAX = 100
        self.HYSTERESIS_THRESHOLD = 50  # Trigger alert level
        self.HYSTERESIS_INC = 10        # Gain per frame (50/10 = 5 frames to trigger)
        self.HYSTERESIS_DEC = 2         # Decay per frame (slower decay for stability)
        
        self.cooldowns = {}
        self.hysteresis_scores = {}  # (track_id, type) -> score

        self.active_track_ids = set()
        self.last_display_data = []

        # Motion Vector Tracking
        self.prev_positions = {}  # {track_id: (x, y, timestamp)}
        self.velocities = {}      # {track_id: (vx, vy)}

        # Scene Memory (Static Object Suppression)
        self.static_candidates = {}  # {id: {start_pos, start_time}}
        self.static_objects = set()  # Set of track IDs confirmed as static noise

        # Sticky Association (Tracking Continuity)
        self.last_matches = {}     # {person_id: phone_id}
        self.match_age = {}        # {person_id: frames_since_last_match}

    def _update_velocities(self, track_id, cx, cy, current_time):
        """
        Update velocity vector for a track ID.

        FIXED: More responsive EMA (90% new, 10% old) for better motion detection.
        """
        if track_id in self.prev_positions:
            px, py, ptime = self.prev_positions[track_id]
            dt = current_time - ptime
            if dt > 0:
                vx = (cx - px) / dt
                vy = (cy - py) / dt

                # FIXED: Less aggressive smoothing (90% new, 10% old)
                if track_id in self.velocities:
                    old_vx, old_vy = self.velocities[track_id]
                    vx = 0.9 * vx + 0.1 * old_vx  # More responsive
                    vy = 0.9 * vy + 0.1 * old_vy

                self.velocities[track_id] = (vx, vy)

        self.prev_positions[track_id] = (cx, cy, current_time)

    def _get_or_create_phone_pseudo_id(self, bbox, existing_ids, current_time):
        """
        FIXED: Create pseudo-ID for phones without persistent tracking IDs.

        Uses spatial hashing to maintain phone identity across frames.
        """
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        # Check if any existing phone ID is nearby (within 50px)
        for pid in existing_ids:
            if pid in self.prev_positions:
                px, py, ptime = self.prev_positions[pid]
                if math.hypot(cx - px, cy - py) < 50:
                    return pid

        # Create new pseudo-ID based on spatial location
        pseudo_id = 1000000 + hash((int(cx / 50), int(cy / 50))) % 1000000
        return pseudo_id

    def process_frame(self, frame, frame_count, skip_frames=5, save_screenshots=True,
                     conf_threshold=0.25, camera_name="Unknown"):
        """
        Process frame with all advanced optimizations.
        """
        current_time = time.time()
        
        # Periodic Cleanup (Every 100 frames = ~3 seconds at 30fps)
        if frame_count % 100 == 0:
            # Clean old cooldowns
            cutoff_time = current_time - (self.COOLDOWN_SECONDS * 2)
            self.cooldowns = {k: v for k, v in self.cooldowns.items() if v > cutoff_time}

            # Clean hysteresis for inactive IDs
            self.hysteresis_scores = {k: v for k, v in self.hysteresis_scores.items()
                                    if k[0] in self.active_track_ids}

            # FIXED: Don't wipe all tracking state every 1000 frames
            # Instead, clean old entries intelligently
            if frame_count % 1000 == 0:
                # Clean velocity/position history for tracks not seen in 60 seconds
                pos_cutoff = current_time - 60
                old_track_ids = [k for k, v in self.prev_positions.items() if v[2] < pos_cutoff]

                for tid in old_track_ids:
                    self.prev_positions.pop(tid, None)
                    self.velocities.pop(tid, None)
                    self.match_age.pop(tid, None)

                # Clean static candidates not seen recently
                for pid in list(self.static_candidates.keys()):
                    if pid not in self.prev_positions:
                        del self.static_candidates[pid]
                        self.static_objects.discard(pid)

                # Don't clear active_track_ids - let frame updates handle it
                # self.active_track_ids.clear()  # REMOVED - was causing 33-second resets

        global_status = "safe"
        screenshot_saved_global = False

        if frame_count % skip_frames == 0:
            self.last_display_data = []
            
            # ==========================================
            # STEP 1: YOLO Inference (Detection + Tracking)
            # ==========================================
            classes_to_track = [self.PERSON_CLASS_ID, self.PHONE_CLASS_ID]
            try:
                results = self.model.track(frame, classes=classes_to_track, conf=conf_threshold,
                                         persist=True, verbose=False, imgsz=1280)
            except Exception as e:
                print(f"[{camera_name}] YOLO error: {e}")
                results = []
            
            person_boxes = []  # (x1, y1, x2, y2, track_id)
            phone_boxes = []   # (x1, y1, x2, y2, conf, track_id)

            current_frame_phone_ids = set()

            if len(results) > 0 and results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    coords = box.xyxy[0].cpu().numpy()

                    # Get or create track ID
                    if box.id is not None:
                        track_id = int(box.id.item())
                    else:
                        # FIXED: Handle phones without IDs using pseudo-ID
                        if cls_id == self.PHONE_CLASS_ID:
                            track_id = self._get_or_create_phone_pseudo_id(
                                coords, current_frame_phone_ids, current_time
                            )
                        else:
                            continue  # Skip persons without IDs

                    # Update velocity tracking for all objects
                    cx = (coords[0] + coords[2]) / 2
                    cy = (coords[1] + coords[3]) / 2
                    self._update_velocities(track_id, cx, cy, current_time)

                    if cls_id == self.PERSON_CLASS_ID:
                        person_boxes.append((*coords, track_id))
                        self.active_track_ids.add(track_id)
                    elif cls_id == self.PHONE_CLASS_ID:
                        conf = float(box.conf[0].item())
                        phone_boxes.append((*coords, conf, track_id))
                        current_frame_phone_ids.add(track_id)

            # ==========================================
            # STEP 2: Static Scene Memory Update
            # ==========================================
            # FIXED: Build lookup dict for O(1) access instead of O(N) search
            phone_dict = {p[5]: p for p in phone_boxes}

            # Check existing candidates
            for pid in list(self.static_candidates.keys()):
                if pid not in current_frame_phone_ids:
                    # Lost tracking
                    del self.static_candidates[pid]
                    self.static_objects.discard(pid)
                    continue

                # Get current position
                curr_box = phone_dict.get(pid)
                if curr_box:
                    cx = (curr_box[0] + curr_box[2]) / 2
                    cy = (curr_box[1] + curr_box[3]) / 2

                    cand = self.static_candidates[pid]
                    dist = math.hypot(cx - cand['start_pos'][0], cy - cand['start_pos'][1])
                    duration = current_time - cand['start_time']

                    # FIXED: Check duration FIRST, then movement
                    if dist > 10:  # FIXED: Was 20px, now 10px (tighter threshold)
                        # Moved too much - reset
                        del self.static_candidates[pid]
                        self.static_objects.discard(pid)
                    elif duration > 3.0:  # Static for 3+ seconds
                        self.static_objects.add(pid)

            # Add new candidates
            for p in phone_boxes:
                pid = p[5]
                if pid not in self.static_candidates and pid not in self.static_objects:
                    cx = (p[0] + p[2]) / 2
                    cy = (p[1] + p[3]) / 2
                    self.static_candidates[pid] = {
                        'start_pos': (cx, cy),
                        'start_time': current_time
                    }

            # ==========================================
            # STEP 3: Pose Inference
            # ==========================================
            pose_keypoints_map = {}
            if hasattr(self.sleep_detector, 'pose_model'):
                try:
                    pose_results = self.sleep_detector.pose_model(frame, verbose=False, conf=0.5)
                    if len(pose_results) > 0 and pose_results[0].boxes:
                        pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                        pose_kpts = pose_results[0].keypoints.xy.cpu().numpy()
                        pose_keypoints_map = self._associate_pose_to_persons(
                            person_boxes, pose_boxes, pose_kpts
                        )
                except Exception as e:
                    print(f"[{camera_name}] Pose inference error: {e}")

            # ==========================================
            # STEP 4: Advanced Phone-Person Association
            # ==========================================
            phone_map = self._associate_phones_to_persons_advanced(
                person_boxes, phone_boxes, pose_keypoints_map, current_time
            )

            # ==========================================
            # STEP 5: Per-Person Status & Hysteresis
            # ==========================================
            for p_box in person_boxes:
                x1, y1, x2, y2, track_id = map(int, p_box)

                # Current frame detection
                is_texting = phone_map.get(track_id, False)
                is_sleeping = False

                # Sleep detection (if not texting)
                if not is_texting:
                    h, w, _ = frame.shape
                    pad = 20
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(w, x2 + pad)
                    cy2 = min(h, y2 + pad)
                    person_crop = frame[cy1:cy2, cx1:cx2]
                    
                    if person_crop.size > 0:
                        sleep_key = f"id_{track_id}"
                        kpts = pose_keypoints_map.get(track_id)
                        status_str, sleep_meta = self.sleep_detector.process_crop(
                            person_crop, id_key=sleep_key, keypoints=kpts, crop_origin=(cx1, cy1)
                        )
                        if status_str == "sleeping":
                            is_sleeping = True

                # ==========================================
                # Bucket-Brigade Hysteresis Update
                # ==========================================
                # Texting score
                t_key = (track_id, "texting")
                t_score = self.hysteresis_scores.get(t_key, 0)
                if is_texting:
                    t_score = min(self.HYSTERESIS_MAX, t_score + self.HYSTERESIS_INC)
                else:
                    t_score = max(0, t_score - self.HYSTERESIS_DEC)
                self.hysteresis_scores[t_key] = t_score

                # Sleeping score
                s_key = (track_id, "sleeping")
                s_score = self.hysteresis_scores.get(s_key, 0)
                if is_sleeping:
                    s_score = min(self.HYSTERESIS_MAX, s_score + self.HYSTERESIS_INC)
                else:
                    s_score = max(0, s_score - self.HYSTERESIS_DEC)
                self.hysteresis_scores[s_key] = s_score

                # ==========================================
                # Determine Final Status
                # ==========================================
                final_status = "safe"
                color = (0, 255, 0)
                label = f"ID: {track_id}"

                # Priority: Texting > Sleeping
                if t_score >= self.HYSTERESIS_THRESHOLD:
                    final_status = "texting"
                    color = (0, 0, 255)
                    global_status = "texting"
                    label += f" [PHONE {int(t_score)}%]"
                elif s_score >= self.HYSTERESIS_THRESHOLD:
                    final_status = "sleeping"
                    color = (255, 0, 0)
                    if global_status != "texting":
                        global_status = "sleeping"
                    label += f" [SLEEP {int(s_score)}%]"

                # ==========================================
                # Evidence Saving
                # ==========================================
                if save_screenshots and final_status != "safe":
                    key = (track_id, final_status)
                    last_time = self.cooldowns.get(key, 0)

                    # Only save when score is saturated (>80) to avoid flickering saves
                    if t_score > 80 or s_score > 80:
                        if (current_time - last_time) > self.COOLDOWN_SECONDS:
                            self.save_evidence(
                                frame, x1, y1, x2, y2, camera_name,
                                "PHONE" if final_status == "texting" else "SLEEP",
                                track_id
                            )
                            screenshot_saved_global = True
                            self.cooldowns[key] = current_time

                self.last_display_data.append((x1, y1, x2, y2, color, final_status, label))

        # ==========================================
        # Drawing
        # ==========================================
        for (x1, y1, x2, y2, color, status, label) in self.last_display_data:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if status != "safe":
                status_text = "PHONE" if status == "texting" else status.upper()
                cv2.putText(frame, status_text, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame, global_status, screenshot_saved_global

    def _associate_phones_to_persons_advanced(self, person_boxes, phone_boxes,
                                             pose_keypoints_map, current_time):
        """
        FIXED: Advanced association with all 5 optimizations properly implemented.

        Features:
        1. Scale-invariant cost calculation
        2. Motion vector alignment (FIXED: strong penalty)
        3. Static object suppression (FIXED: proper timing)
        4. Sticky association (FIXED: decay on occlusion)
        5. Elbow angle validation (FIXED: check correct arm, strong penalty)
        """
        mapping = {}
        if not phone_boxes or not person_boxes:
            return mapping

        # FIXED: Add safety check for tuple length
        if len(phone_boxes[0]) != 6:
            print(f"WARNING: Expected phone_boxes with 6 elements, got {len(phone_boxes[0])}")
            return mapping

        cost_matrix = []
        new_matches = {}

        for person_bbox in person_boxes:
            p_x1, p_y1, p_x2, p_y2, p_id = person_bbox
            person_width = p_x2 - p_x1
            person_height = p_y2 - p_y1
            p_cx = (p_x1 + p_x2) / 2
            p_chest_y = p_y1 + person_height * 0.4

            # Get person velocity
            v_person = self.velocities.get(p_id, (0, 0))
            vp_mag = math.hypot(v_person[0], v_person[1])

            row_costs = []

            for phone_bbox in phone_boxes:
                ph_x1, ph_y1, ph_x2, ph_y2, conf, ph_id = phone_bbox
                ph_cx = (ph_x1 + ph_x2) / 2
                ph_cy = (ph_y1 + ph_y2) / 2

                # ==========================================
                # Base Cost (Scale-Invariant)
                # ==========================================
                cost = float('inf')

                # Adaptive padding
                pad_x = person_width * 0.4
                pad_y = person_height * 0.3

                if (p_x1 - pad_x <= ph_cx <= p_x2 + pad_x and
                    p_y1 - pad_y <= ph_cy <= p_y2 + pad_y):

                    kpts = pose_keypoints_map.get(p_id)
                    raw_dist = float('inf')

                    # Try wrist distance first
                    if kpts is not None:
                        wrists = []
                        if kpts[9][0] > 0: wrists.append(kpts[9])
                        if kpts[10][0] > 0: wrists.append(kpts[10])

                        for w in wrists:
                            d = math.hypot(ph_cx - w[0], ph_cy - w[1])
                            if d < raw_dist:
                                raw_dist = d

                    # Fallback to chest distance
                    if raw_dist == float('inf'):
                        raw_dist = math.hypot(ph_cx - p_cx, ph_cy - p_chest_y)

                    # Normalize by person height
                    cost = raw_dist / person_height

                    # ==========================================
                    # Optimization 1: Sticky Association
                    # ==========================================
                    if self.last_matches.get(p_id) == ph_id:
                        cost *= 0.6  # 40% discount for continuity

                    # ==========================================
                    # Optimization 2: Static Object Suppression
                    # ==========================================
                    if ph_id in self.static_objects:
                        cost += 5.0  # Strong penalty for known static objects

                        # EXCEPTION: Pickup detection (wrist close)
                        # FIXED: Increased threshold from 5% to 15% to catch stationary phones in hand
                        if raw_dist < person_height * 0.15:
                            cost -= 5.0  # Cancel penalty

                    # ==========================================
                    # Optimization 3: Motion Vector Alignment
                    # ==========================================
                    if vp_mag > 2.0:  # Person is moving
                        v_phone = self.velocities.get(ph_id, (0, 0))
                        v_ph_mag = math.hypot(v_phone[0], v_phone[1])

                        # FIXED: Strong penalty (was 2.0, now 100.0)
                        if v_ph_mag < 0.5:  # Phone is stationary
                            cost += 100.0  # Reject - person moving, phone still
                        else:
                            # Check direction alignment
                            vp_norm = (v_person[0] / vp_mag, v_person[1] / vp_mag)
                            vph_norm = (v_phone[0] / v_ph_mag, v_phone[1] / v_ph_mag)
                            dot = vp_norm[0] * vph_norm[0] + vp_norm[1] * vph_norm[1]

                            if dot < 0:  # Moving in opposite directions
                                cost += 100.0  # Strong rejection

                    # ==========================================
                    # Optimization 4: Elbow Angle Validation
                    # ==========================================
                    if kpts is not None:
                        # FIXED: Determine which wrist is closer to phone
                        left_wrist = kpts[9]
                        right_wrist = kpts[10]

                        dist_left = math.hypot(ph_cx - left_wrist[0], ph_cy - left_wrist[1])
                        dist_right = math.hypot(ph_cx - right_wrist[0], ph_cy - right_wrist[1])

                        # Check only the arm that's actually near the phone
                        if dist_left < dist_right and left_wrist[0] > 0:
                            arm_to_check = (5, 7, 9)  # Left: shoulder, elbow, wrist
                        elif right_wrist[0] > 0:
                            arm_to_check = (6, 8, 10)  # Right: shoulder, elbow, wrist
                        else:
                            arm_to_check = None

                        if arm_to_check:
                            s_idx, e_idx, w_idx = arm_to_check

                            if (kpts[s_idx][0] > 0 and kpts[e_idx][0] > 0 and
                                kpts[w_idx][0] > 0):
                                # Calculate elbow angle
                                a = kpts[s_idx]
                                b = kpts[e_idx]
                                c = kpts[w_idx]

                                ba = a - b
                                bc = c - b

                                dot_prod = np.dot(ba, bc)
                                mag_prod = np.linalg.norm(ba) * np.linalg.norm(bc)

                                if mag_prod > 0:
                                    cosine_angle = dot_prod / mag_prod
                                    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
                                    angle = np.degrees(np.arccos(cosine_angle))

                                    # FIXED: Stronger penalty (was 1.0, now 5.0)
                                    # Texting requires bent elbow (< 120°)
                                    if angle > 150:  # Arm too straight
                                        cost += 5.0  # Strong penalty

                                    # BIOMECHANICAL OVERRIDE for Static Phones
                                    # If arm is bent (<120) and wrist is reasonably close (<25%),
                                    # we assume they are using a stationary phone (e.g., on desk)
                                    if angle < 120 and raw_dist < person_height * 0.25:
                                        if ph_id in self.static_objects:
                                            # If we haven't already cancelled the penalty in step 2
                                            if raw_dist >= person_height * 0.15: # If step 2 failed
                                                cost -= 5.0 # Cancel static penalty

                    # ==========================================
                    # Overhead Phone Penalty
                    # ==========================================
                    if ph_cy < p_y1:  # Phone above person's head
                        cost += 100.0  # Strong rejection

                row_costs.append(cost)
            cost_matrix.append(row_costs)

        # ==========================================
        # Hungarian Algorithm
        # ==========================================
        C = np.array(cost_matrix)
        C[C == float('inf')] = 100000

        try:
            row_ind, col_ind = linear_sum_assignment(C)
        except Exception as e:
            print(f"Hungarian algorithm error: {e}")
            return mapping

        # Filter by threshold
        MAX_NORM_COST = 0.6

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r][c] < MAX_NORM_COST:
                p_id = person_boxes[r][4]
                ph_id = phone_boxes[c][5]
                mapping[p_id] = True
                new_matches[p_id] = ph_id

        # ==========================================
        # FIXED: Sticky Association with Decay
        # ==========================================
        # Don't immediately wipe old matches - use decay counter
        for p_id in list(self.last_matches.keys()):
            if p_id not in new_matches:
                # Increment age counter
                self.match_age[p_id] = self.match_age.get(p_id, 0) + 1

                # Remove only after 10 frames without match (occlusion tolerance)
                if self.match_age[p_id] > 10:
                    del self.last_matches[p_id]
                    self.match_age.pop(p_id, None)
            else:
                # Reset age counter
                self.match_age[p_id] = 0

        # Update with new matches
        self.last_matches.update(new_matches)

        return mapping

    def _associate_pose_to_persons(self, person_boxes, pose_boxes, pose_kpts):
        """
        Associate pose detections with person detections using IoU + center fallback.
        """
        mapping = {}
        if len(person_boxes) == 0 or len(pose_boxes) == 0:
            return mapping
        
        for p_box in person_boxes:
            px1, py1, px2, py2, track_id = p_box
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
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = p_area + (pox2 - pox1) * (poy2 - poy1) - inter
                    iou = inter / union if union > 0 else 0

                # Center point fallback
                po_cx = (pox1 + pox2) / 2
                po_cy = (poy1 + poy2) / 2
                center_inside = (px1 <= po_cx <= px2) and (py1 <= po_cy <= py2)

                # Custom scoring
                score = iou
                if center_inside and score < 0.4:
                    score = 0.4  # Boost if center matches

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx != -1 and best_score > 0.3:
                mapping[track_id] = pose_kpts[best_idx]

        return mapping

    def save_evidence(self, frame, x1, y1, x2, y2, camera_name, type_str, track_id):
        """Save evidence screenshot with metadata."""
        evidence_img = frame.copy()
        color = (0, 0, 255) if type_str == "PHONE" else (255, 0, 0)
        cv2.rectangle(evidence_img, (x1, y1), (x2, y2), color, 3)

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(evidence_img, (0, 0), (evidence_img.shape[1], 40), (0, 0, 0), -1)
        header_text = f"{type_str} | {camera_name} | {ts} | ID: {track_id}"
        cv2.putText(evidence_img, header_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        timestamp_fn = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        safe_cam_name = "".join([c for c in camera_name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        filename = os.path.join(
            self.output_dir,
            f"evidence_{type_str.lower()}_{safe_cam_name}_{timestamp_fn}_id{track_id}.jpg"
        )
        cv2.imwrite(filename, evidence_img)
        print(f"📸 EVIDENCE SAVED: {filename}")
