import cv2
import time
import os
import math
import datetime
from ultralytics import YOLO
import threading
from sleep_detector import SleepDetector

class PhoneDetector:
    def __init__(self, model_path='yolo26n.pt', output_dir="detections", cooldown_seconds=5, consistency_threshold=3, model_instance=None, pose_model_instance=None, lock=None):
        
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
        
        # Temporal consistency: list of {'center': (x,y), 'streak': int, 'last_seen': time, 'id': int, 'cooldowns': {}}
        self.detection_streaks = []
        self.CONSISTENCY_THRESHOLD = consistency_threshold
        self.SLEEP_CONSISTENCY_THRESHOLD = 3

        # Simulated Person Tracking ID
        self.next_person_id = 1
        
        # State for frame skipping
        self.last_display_data = [] # List of (x1, y1, x2, y2, color, status, text)

    def process_frame(self, frame, frame_count, skip_frames=5, save_screenshots=True, conf_threshold=0.25, camera_name="Unknown"):
        """
        Process the frame.
        skip_frames: Run heavy inference only every N frames.
        Returns: frame, global_status_string, screenshot_saved_bool
        """
        current_time = time.time()
        
        global_status = "safe" # safe, texting, sleeping
        screenshot_saved_global = False

        # --- HEAVY INFERENCE STEP ---
        if frame_count % skip_frames == 0:
            self.last_display_data = [] # Reset display data
            
            # 1. First Pass: Find People in the full room
            # Thread-safe inference
            if self.lock:
                with self.lock:
                    results = self.model.predict(frame, classes=[self.PERSON_CLASS_ID], conf=conf_threshold, verbose=False)
            else:
                results = self.model.predict(frame, classes=[self.PERSON_CLASS_ID], conf=conf_threshold, verbose=False)
            
            if len(results) > 0:
                boxes = results[0].boxes
                
                person_crops = []
                person_metadata = []

                # Collect crops and resolve tracking
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    p_cx, p_cy = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # --- ZOOM LOGIC ---
                    h, w, _ = frame.shape
                    pad = 20
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(w, x2 + pad)
                    cy2 = min(h, y2 + pad)
                    
                    person_crop = frame[cy1:cy2, cx1:cx2]
                    
                    if person_crop.size > 0:
                        person_crops.append(person_crop)

                        # --- TRACKING: Resolve ID immediately ---
                        matched_candidate = None
                        for candidate in self.detection_streaks:
                            lx, ly = candidate['center']
                            dist = math.sqrt((p_cx - lx)**2 + (p_cy - ly)**2)
                            if dist < 100:
                                matched_candidate = candidate
                                candidate['center'] = (p_cx, p_cy)
                                candidate['last_seen'] = current_time
                                break

                        if matched_candidate is None:
                            matched_candidate = {
                                'id': self.next_person_id,
                                'center': (p_cx, p_cy),
                                'streak_phone': 0,
                                'streak_sleep': 0,
                                'last_seen': current_time,
                                'cooldowns': {'phone': 0, 'sleep': 0}
                            }
                            self.next_person_id += 1
                            self.detection_streaks.append(matched_candidate)

                        # Store context
                        person_metadata.append({
                            'idx': idx,
                            'box': (x1, y1, x2, y2),
                            'center': (p_cx, p_cy),
                            'candidate': matched_candidate
                        })

                # 2. Second Pass: Run AI on ALL crops in batch (PHONE)
                batch_phone_results = []
                if len(person_crops) > 0:
                    if self.lock:
                        with self.lock:
                            batch_phone_results = self.model.predict(person_crops, classes=[self.PHONE_CLASS_ID], conf=0.15, verbose=False)
                    else:
                        batch_phone_results = self.model.predict(person_crops, classes=[self.PHONE_CLASS_ID], conf=0.15, verbose=False)

                # 3. Third Pass: Identify Sleep Candidates and Batch Sleep
                sleep_crops = []
                sleep_ids = []
                sleep_map_indices = [] # Map index in sleep_results back to person_metadata index

                for i, meta in enumerate(person_metadata):
                    crop_result = batch_phone_results[i]
                    candidate = meta['candidate']

                    # Check Phone
                    has_phone = False
                    if len(crop_result) > 0 and len(crop_result.boxes) > 0:
                        has_phone = True

                    # Update Phone Streak
                    if has_phone:
                        candidate['streak_phone'] += 1
                    else:
                        candidate['streak_phone'] = max(0, candidate['streak_phone'] - 1)
                        
                    # Determine Temporary Status (Texting takes priority)
                    is_texting = candidate['streak_phone'] >= self.CONSISTENCY_THRESHOLD

                    if not is_texting:
                        # Candidate for Sleep Detection
                        sleep_crops.append(person_crops[i])
                        sleep_ids.append(f"{camera_name}_pid_{candidate['id']}")
                        sleep_map_indices.append(i)

                # Batch Sleep Inference
                batch_sleep_results = []
                if sleep_crops:
                    batch_sleep_results = self.sleep_detector.process_batch(sleep_crops, sleep_ids)

                # 4. Final Pass: Merge Results and Update State
                for i, meta in enumerate(person_metadata):
                    candidate = meta['candidate']
                    x1, y1, x2, y2 = meta['box']

                    status = "safe"
                    color = (0, 255, 0) # Green
                    display_text = ""

                    # --- PHONE LOGIC ---
                    if candidate['streak_phone'] >= self.CONSISTENCY_THRESHOLD:
                        status = "texting"
                        color = (0, 0, 255) # Red
                        global_status = "texting"
                        display_text = f"P{candidate['id']}: PHONE"

                        # Cooldown Save
                        if save_screenshots:
                            last_shot = candidate['cooldowns']['phone']
                            if (current_time - last_shot) > 120:
                                candidate['cooldowns']['phone'] = current_time
                                self.save_evidence(frame, x1, y1, x2, y2, camera_name, "PHONE", candidate['id'])
                                screenshot_saved_global = True

                        # Reset Sleep Streak if Texting
                        candidate['streak_sleep'] = max(0, candidate['streak_sleep'] - 1)

                    # --- SLEEP LOGIC ---
                    else:
                        # Retrieve result from batch if it exists
                        # Find if this 'i' was in the sleep map
                        try:
                            s_idx = sleep_map_indices.index(i)
                            sleep_res = batch_sleep_results[s_idx]

                            if sleep_res:
                                sleep_status, sleep_info = sleep_res

                                if sleep_status == "sleeping":
                                    candidate['streak_sleep'] += 1

                                    if candidate['streak_sleep'] >= self.SLEEP_CONSISTENCY_THRESHOLD:
                                        status = "sleeping"
                                        color = (255, 0, 0) # Blue
                                        if global_status != "texting":
                                            global_status = "sleeping"
                                        display_text = f"P{candidate['id']}: SLEEP"

                                        if save_screenshots:
                                            last_shot = candidate['cooldowns']['sleep']
                                            if (current_time - last_shot) > 120:
                                                candidate['cooldowns']['sleep'] = current_time
                                                self.save_evidence(frame, x1, y1, x2, y2, camera_name, "SLEEP", candidate['id'])
                                                screenshot_saved_global = True
                                else:
                                    candidate['streak_sleep'] = max(0, candidate['streak_sleep'] - 1)
                                    if sleep_status == "drowsy":
                                        color = (0, 255, 255) # Yellow
                            else:
                                 candidate['streak_sleep'] = max(0, candidate['streak_sleep'] - 1)
                        except ValueError:
                            # Not in sleep list (shouldn't happen if logic above holds, but safe fallback)
                            pass

                    self.last_display_data.append((x1, y1, x2, y2, color, status, display_text))

            # --- PRUNING STREAKS ---
            self.detection_streaks = [
                d for d in self.detection_streaks 
                if (current_time - d['last_seen']) < 1.0
            ]

        # --- DRAWING ---
        for (x1, y1, x2, y2, color, status, text) in self.last_display_data:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if text:
                 cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if status == "texting":
                global_status = "texting"
            elif status == "sleeping" and global_status != "texting":
                global_status = "sleeping"

        return frame, global_status, screenshot_saved_global

    def save_evidence(self, frame, x1, y1, x2, y2, camera_name="Unknown", detection_type="PHONE", person_id=0):
        """
        Save evidence screenshot.
        
        Args:
            detection_type: "PHONE" or "SLEEP"
        """
        evidence_img = frame.copy()
        
        # Draw the box on the evidence
        box_color = (0, 0, 255) if detection_type == "PHONE" else (255, 0, 0)  # Red for phone, Blue for sleep
        cv2.rectangle(evidence_img, (x1, y1), (x2, y2), box_color, 3)
        
        # Add Header
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Draw black bar at top
        cv2.rectangle(evidence_img, (0, 0), (evidence_img.shape[1], 40), (0,0,0), -1)
        # Add text
        header_text = f"{detection_type} | {camera_name} | P{person_id} | {ts}"
        cv2.putText(evidence_img, header_text, (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        timestamp_fn = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Sanitize camera name for filename
        safe_cam_name = "".join([c for c in camera_name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        
        # Add detection type to filename
        filename = os.path.join(self.output_dir, f"evidence_{detection_type.lower()}_{safe_cam_name}_p{person_id}_{timestamp_fn}.jpg")
        cv2.imwrite(filename, evidence_img)
        print(f"📸 EVIDENCE SAVED: {filename}")
