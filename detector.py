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

        # Replaced simple consistency_threshold with Bucket Hysteresis config
        self.HYSTERESIS_MAX = 100
        self.HYSTERESIS_THRESHOLD = 50  # Trigger alert level
        self.HYSTERESIS_INC = 10        # Gain per frame
        self.HYSTERESIS_DEC = 2         # Decay per frame
        
        self.cooldowns = {}
        self.hysteresis_scores = {} # (track_id, type) -> score

        self.active_track_ids = set()
        self.last_display_data = []

        # --- NEW STATE TRACKING ---
        self.prev_positions = {}  # {track_id: (x, y, time)}
        self.velocities = {}      # {track_id: (vx, vy)}

        self.static_candidates = {} # {id: {start_pos, start_time, last_seen}}
        self.static_objects = set() # Set of track IDs confirmed as static noise

        self.last_matches = {}    # {person_id: phone_id} for sticky tracking

    def _update_velocities(self, track_id, cx, cy, current_time):
        """Update velocity vector for a track ID."""
        if track_id in self.prev_positions:
            px, py, ptime = self.prev_positions[track_id]
            dt = current_time - ptime
            if dt > 0:
                vx = (cx - px) / dt
                vy = (cy - py) / dt

                # Smooth velocity (Exponential Moving Average)
                if track_id in self.velocities:
                    old_vx, old_vy = self.velocities[track_id]
                    vx = 0.7 * vx + 0.3 * old_vx
                    vy = 0.7 * vy + 0.3 * old_vy

                self.velocities[track_id] = (vx, vy)

        self.prev_positions[track_id] = (cx, cy, current_time)

    def _update_static_scene_memory(self, phone_boxes, current_time):
        """
        Track unassociated phones to identify static noise (wall clocks, stickers).
        Logic: If phone stays within 10px radius for > 3 seconds, mark as static.
        """
        active_ids = set()

        # This function assumes 'phone_boxes' has tracking IDs if possible.
        # However, standard YOLO track() gives IDs. If detection only, we can't do ID-based static logic easily.
        # But we are using .track(), so phones *should* have IDs if persist=True.
        # Let's assume phone_boxes contains ID if available.
        # The main loop unpacks 5 values usually. We might need to adjust unpacking.

        # NOTE: Ultralytics track() returns boxes with IDs.
        # But in process_frame we unpack: x1, y1, x2, y2, conf.
        # We need to preserve ID for Scene Memory.
        # Modified process_frame to pass phone IDs.

        pass # Implemented inside process_frame loop for easier data access

    def process_frame(self, frame, frame_count, skip_frames=5, save_screenshots=True,
                     conf_threshold=0.25, camera_name="Unknown"):
        current_time = time.time()
        
        # Periodic Cleanup
        if frame_count % 100 == 0:
            cutoff_time = current_time - (self.COOLDOWN_SECONDS * 2)
            self.cooldowns = {k: v for k, v in self.cooldowns.items() if v > cutoff_time}

            # Clean hysteresis for inactive IDs
            self.hysteresis_scores = {k: v for k, v in self.hysteresis_scores.items()
                                    if k[0] in self.active_track_ids}

            # Clean velocity/position history
            if frame_count % 1000 == 0:
                self.active_track_ids.clear()
                self.prev_positions.clear()
                self.velocities.clear()
                self.static_candidates.clear()
                self.static_objects.clear()
                self.last_matches.clear()

        global_status = "safe"
        screenshot_saved_global = False

        if frame_count % skip_frames == 0:
            self.last_display_data = []
            
            # 1. Inference
            classes_to_track = [self.PERSON_CLASS_ID, self.PHONE_CLASS_ID]
            try:
                results = self.model.track(frame, classes=classes_to_track, conf=conf_threshold,
                                         persist=True, verbose=False, imgsz=1280)
            except Exception as e:
                print(f"[{camera_name}] YOLO error: {e}")
                results = []
            
            person_boxes = [] # (x1, y1, x2, y2, id)
            phone_boxes = []  # (x1, y1, x2, y2, conf, id) <-- Added ID

            current_frame_phone_ids = set()

            if len(results) > 0 and results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    coords = box.xyxy[0].cpu().numpy()

                    # Update Velocity for everything tracked
                    if box.id is not None:
                        track_id = int(box.id.item())
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

            # --- STATIC SCENE MEMORY UPDATE ---
            # Check existing candidates
            for pid in list(self.static_candidates.keys()):
                if pid not in current_frame_phone_ids:
                    del self.static_candidates[pid] # Lost tracking, reset
                    continue

                # Check movement
                # We need the current position of this phone
                # Inefficient to search list, but N is small (phones < 10)
                curr_box = next((p for p in phone_boxes if p[5] == pid), None)
                if curr_box:
                    cx = (curr_box[0] + curr_box[2]) / 2
                    cy = (curr_box[1] + curr_box[3]) / 2

                    cand = self.static_candidates[pid]
                    dist = math.hypot(cx - cand['start_pos'][0], cy - cand['start_pos'][1])

                    if dist > 20: # Moved > 20px
                        del self.static_candidates[pid]
                        if pid in self.static_objects:
                            self.static_objects.remove(pid)
                    elif (current_time - cand['start_time']) > 3.0:
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


            # 2. Pose Inference
            pose_keypoints_map = {}
            if hasattr(self.sleep_detector, 'pose_model'):
                try:
                    pose_results = self.sleep_detector.pose_model(frame, verbose=False, conf=0.5)
                    if len(pose_results) > 0 and pose_results[0].boxes:
                        pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                        pose_kpts = pose_results[0].keypoints.xy.cpu().numpy()
                        pose_keypoints_map = self._associate_pose_to_persons(person_boxes, pose_boxes, pose_kpts)
                except Exception:
                    pass

            # 3. Association (Advanced)
            phone_map = self._associate_phones_to_persons_advanced(
                person_boxes, phone_boxes, pose_keypoints_map
            )

            # 4. Logic & Hysteresis
            for p_box in person_boxes:
                x1, y1, x2, y2, track_id = map(int, p_box)

                # Current Frame Status
                is_texting = phone_map.get(track_id, False)
                is_sleeping = False

                # Sleep Check (if not texting)
                sleep_meta = {}
                if not is_texting:
                    h, w, _ = frame.shape
                    pad = 20
                    cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
                    cx2 = min(w, x2 + pad); cy2 = min(h, y2 + pad)
                    person_crop = frame[cy1:cy2, cx1:cx2]
                    
                    if person_crop.size > 0:
                        sleep_key = f"id_{track_id}"
                        kpts = pose_keypoints_map.get(track_id)
                        status_str, sleep_meta = self.sleep_detector.process_crop(
                            person_crop, id_key=sleep_key, keypoints=kpts, crop_origin=(cx1, cy1)
                        )
                        if status_str == "sleeping": is_sleeping = True

                # --- BUCKET-BRIGADE HYSTERESIS ---
                # Update Texting Score
                t_key = (track_id, "texting")
                t_score = self.hysteresis_scores.get(t_key, 0)
                if is_texting:
                    t_score = min(self.HYSTERESIS_MAX, t_score + self.HYSTERESIS_INC)
                else:
                    t_score = max(0, t_score - self.HYSTERESIS_DEC)
                self.hysteresis_scores[t_key] = t_score

                # Update Sleeping Score
                s_key = (track_id, "sleeping")
                s_score = self.hysteresis_scores.get(s_key, 0)
                if is_sleeping:
                    s_score = min(self.HYSTERESIS_MAX, s_score + self.HYSTERESIS_INC)
                else:
                    s_score = max(0, s_score - self.HYSTERESIS_DEC)
                self.hysteresis_scores[s_key] = s_score

                # --- DETERMINE FINAL STATUS ---
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
                    if global_status != "texting": global_status = "sleeping"
                    label += f" [SLEEP {int(s_score)}%]"

                # Save Evidence
                if save_screenshots and final_status != "safe":
                    key = (track_id, final_status)
                    last_time = self.cooldowns.get(key, 0)
                    # Only save if score is HIGH (saturated) to avoid flickering saves
                    if t_score > 80 or s_score > 80:
                        if (current_time - last_time) > self.COOLDOWN_SECONDS:
                            self.save_evidence(frame, x1, y1, x2, y2, camera_name,
                                             "PHONE" if final_status == "texting" else "SLEEP", track_id)
                            screenshot_saved_global = True
                            self.cooldowns[key] = current_time

                self.last_display_data.append((x1, y1, x2, y2, color, final_status, label))

        # Drawing
        for (x1, y1, x2, y2, color, status, label) in self.last_display_data:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if status != "safe":
                status_text = "PHONE" if status == "texting" else status.upper()
                cv2.putText(frame, status_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame, global_status, screenshot_saved_global

    def _associate_phones_to_persons_advanced(self, person_boxes, phone_boxes, pose_keypoints_map):
        mapping = {}
        if not phone_boxes or not person_boxes:
            return mapping

        cost_matrix = []
        new_matches = {}

        for person_bbox in person_boxes:
            p_x1, p_y1, p_x2, p_y2, p_id = person_bbox
            person_height = p_y2 - p_y1
            p_cx = (p_x1 + p_x2) / 2
            p_chest_y = p_y1 + person_height * 0.4

            # Velocity of Person
            v_person = self.velocities.get(p_id, (0, 0))
            vp_mag = math.hypot(v_person[0], v_person[1])

            row_costs = []
            for phone_bbox in phone_boxes:
                ph_x1, ph_y1, ph_x2, ph_y2, conf, ph_id = phone_bbox
                ph_cx = (ph_x1 + ph_x2) / 2
                ph_cy = (ph_y1 + ph_y2) / 2

                # --- BASE COST (Scale Invariant) ---
                cost = float('inf')

                # 1. Proximity Check
                pad_x = (p_x2 - p_x1) * 0.4
                pad_y = person_height * 0.3
                if (p_x1 - pad_x <= ph_cx <= p_x2 + pad_x and
                    p_y1 - pad_y <= ph_cy <= p_y2 + pad_y):

                    kpts = pose_keypoints_map.get(p_id)
                    raw_dist = float('inf')

                    # Wrist Distance
                    if kpts is not None:
                        wrists = []
                        if kpts[9][0] > 0: wrists.append(kpts[9])
                        if kpts[10][0] > 0: wrists.append(kpts[10])
                        for w in wrists:
                            d = math.hypot(ph_cx - w[0], ph_cy - w[1])
                            if d < raw_dist: raw_dist = d

                    if raw_dist == float('inf'):
                        raw_dist = math.hypot(ph_cx - p_cx, ph_cy - p_chest_y)

                    cost = raw_dist / person_height

                    # --- ADVANCED HEURISTICS ---

                    # 2. Sticky Association (Momentum Bonus)
                    if self.last_matches.get(p_id) == ph_id:
                        cost *= 0.6  # 40% discount to keep lock

                    # 3. Static Suppression
                    if ph_id in self.static_objects:
                        cost += 5.0 # Massive penalty for known static noise
                        # EXCEPTION: If wrist is extremely close (< 5% height), user picked it up
                        if raw_dist < person_height * 0.05:
                            cost -= 5.0 # Cancel penalty

                    # 4. Motion Vector Alignment
                    # If person is moving fast, phone MUST move with them
                    if vp_mag > 2.0: # Arbitrary velocity threshold
                        v_phone = self.velocities.get(ph_id, (0,0))
                        v_ph_mag = math.hypot(v_phone[0], v_phone[1])

                        # Check 1: Phone static while person moves?
                        if v_ph_mag < 0.5:
                            cost += 2.0 # Penalty
                        else:
                            # Check 2: Direction alignment (Dot Product)
                            # Normalize
                            vp_norm = (v_person[0]/vp_mag, v_person[1]/vp_mag)
                            vph_norm = (v_phone[0]/v_ph_mag, v_phone[1]/v_ph_mag)
                            dot = vp_norm[0]*vph_norm[0] + vp_norm[1]*vph_norm[1]

                            if dot < 0: # Moving opposite directions
                                cost += 2.0

                    # 5. Elbow Angle Validation (Biomechanical)
                    if kpts is not None:
                        # Left Arm: 5-7-9, Right Arm: 6-8-10
                        # We only check the arm closest to the phone
                        # This is complex, simplifying to: if EITHER arm is straight, apply penalty?
                        # No, valid to have one arm down and one up.
                        # Check arm closest to phone.

                        arms = [
                            (5, 7, 9), # Left
                            (6, 8, 10) # Right
                        ]

                        valid_pose_found = False
                        pose_checked = False

                        for s_idx, e_idx, w_idx in arms:
                            if kpts[s_idx][0] > 0 and kpts[e_idx][0] > 0 and kpts[w_idx][0] > 0:
                                # Calculate Angle
                                a = kpts[s_idx]
                                b = kpts[e_idx]
                                c = kpts[w_idx]

                                ba = a - b
                                bc = c - b

                                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                                angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

                                pose_checked = True
                                # Texting usually < 120 degrees
                                if angle < 120:
                                    valid_pose_found = True

                        # If we successfully checked at least one arm, and NONE were valid (<120)
                        # imply arms are straight down/out.
                        if pose_checked and not valid_pose_found:
                            cost += 1.0 # Penalty for non-texting posture

                    # 6. Overhead Penalty
                    if ph_cy < p_y1:
                        cost += 100.0

                row_costs.append(cost)
            cost_matrix.append(row_costs)

        C = np.array(cost_matrix)
        C[C == float('inf')] = 100000

        try:
            row_ind, col_ind = linear_sum_assignment(C)
        except:
            return mapping

        MAX_NORM_COST = 0.6

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r][c] < MAX_NORM_COST:
                p_id = person_boxes[r][4]
                ph_id = phone_boxes[c][5]
                mapping[p_id] = True
                new_matches[p_id] = ph_id

        self.last_matches = new_matches
        return mapping

    def _associate_pose_to_persons(self, person_boxes, pose_boxes, pose_kpts):
        mapping = {}
        if len(person_boxes) == 0 or len(pose_boxes) == 0:
            return mapping
        
        for p_box in person_boxes:
            px1, py1, px2, py2, track_id = p_box
            best_score = 0
            best_idx = -1

            for i, (pox1, poy1, pox2, poy2) in enumerate(pose_boxes):
                ix1 = max(px1, pox1); iy1 = max(py1, poy1)
                ix2 = min(px2, pox2); iy2 = min(py2, poy2)
                iou = 0
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1)*(iy2-iy1)
                    union = ((px2-px1)*(py2-py1)) + ((pox2-pox1)*(poy2-poy1)) - inter
                    iou = inter/union

                po_cx = (pox1+pox2)/2; po_cy = (poy1+poy2)/2
                center_inside = (px1<=po_cx<=px2) and (py1<=po_cy<=py2)

                score = iou
                if center_inside and score < 0.4: score = 0.4

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx != -1 and best_score > 0.3:
                mapping[track_id] = pose_kpts[best_idx]
        return mapping

    def save_evidence(self, frame, x1, y1, x2, y2, camera_name, type_str, track_id):
        # Implementation identical to previous, just cleaner
        evidence_img = frame.copy()
        color = (0,0,255) if type_str=="PHONE" else (255,0,0)
        cv2.rectangle(evidence_img, (x1,y1), (x2,y2), color, 3)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(evidence_img, f"{type_str} | {camera_name} | {ts}", (10,30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        fn = os.path.join(self.output_dir, f"evidence_{type_str}_{time.time()}.jpg")
        cv2.imwrite(fn, evidence_img)
        print(f"EVIDENCE SAVED: {fn}")
