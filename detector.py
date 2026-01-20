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

        # Sleep Detection Tracking (Similar to phone detection)
        # Note: We merge sleep tracking into the main person tracking via detection_streaks to keep IDs consistent if possible
        # But for now, we keep sleep streaks separate if they are purely based on crop analysis,
        # however, to satisfy "Person ID" requirement, we should ideally link them.
        # Given the "batched inference" structure, we are iterating over PERSON crops.
        # So we can track EVERYTHING on the `detection_streaks` object which represents a "Person".
        
        # Removed: self.sleep_streaks (Now we track sleep on the person object)
        
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
            
            # Temporary list to track who we saw this frame (for streak management)
            current_frame_detections = [] 
            
            # 1. First Pass: Find People in the full room
            # We filter for PERSON class (0) only
            
            # Thread-safe inference
            if self.lock:
                with self.lock:
                    results = self.model.predict(frame, classes=[self.PERSON_CLASS_ID], conf=conf_threshold, verbose=False)
            else:
                results = self.model.predict(frame, classes=[self.PERSON_CLASS_ID], conf=conf_threshold, verbose=False)
            
            # Use cpu().numpy() or .tolist() to get coordinates
            if len(results) > 0:
                boxes = results[0].boxes
                
                # --- NEW BATCHING LOGIC ---
                person_crops = []
                person_metadata = []

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
                        # Store context needed for loop
                        person_metadata.append({
                            'idx': idx,
                            'box': (x1, y1, x2, y2),
                            'center': (p_cx, p_cy)
                        })

                # 2. Second Pass: Run AI on ALL crops in batch
                # Look for PHONE class (67) with lower threshold
                batch_results = []
                if len(person_crops) > 0:
                    # Thread-safe inference
                    if self.lock:
                        with self.lock:
                            batch_results = self.model.predict(person_crops, classes=[self.PHONE_CLASS_ID], conf=0.15, verbose=False)
                    else:
                        batch_results = self.model.predict(person_crops, classes=[self.PHONE_CLASS_ID], conf=0.15, verbose=False)

                # Process Results
                for i, meta in enumerate(person_metadata):
                    idx = meta['idx']
                    x1, y1, x2, y2 = meta['box']
                    p_cx, p_cy = meta['center']
                    person_crop = person_crops[i]
                    crop_result = batch_results[i]

                    # Default status
                    status = "safe"
                    color = (0, 255, 0) # Green
                    display_text = ""

                    # Check Phone Detection
                    has_phone = False
                    if len(crop_result) > 0 and len(crop_result.boxes) > 0:
                        has_phone = True

                    # --- TEMPORAL CONSISTENCY & TRACKING ---
                    # Match with existing streak (simulated tracking)
                    matched_candidate = None
                    for candidate in self.detection_streaks:
                        lx, ly = candidate['center']
                        dist = math.sqrt((p_cx - lx)**2 + (p_cy - ly)**2)

                        # Match if within 100 pixels
                        if dist < 100:
                            matched_candidate = candidate
                            # Update Position
                            candidate['center'] = (p_cx, p_cy)
                            candidate['last_seen'] = current_time
                            break
                    
                    if matched_candidate is None:
                        # New Person
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

                    # Mark as processed
                    # (We don't need a separate list if we update last_seen)

                    # --- LOGIC: PHONE ---
                    if has_phone:
                        matched_candidate['streak_phone'] += 1
                    else:
                        matched_candidate['streak_phone'] = max(0, matched_candidate['streak_phone'] - 1)
                        
                    # Check Phone Threshold
                    if matched_candidate['streak_phone'] >= self.CONSISTENCY_THRESHOLD:
                        status = "texting"
                        color = (0, 0, 255) # Red
                        global_status = "texting"
                        display_text = f"P{matched_candidate['id']}: PHONE"

                        # Check Cooldown
                        if save_screenshots:
                            last_shot = matched_candidate['cooldowns']['phone']
                            if (current_time - last_shot) > 120: # 120s Cooldown
                                matched_candidate['cooldowns']['phone'] = current_time
                                self.save_evidence(frame, x1, y1, x2, y2, camera_name, "PHONE", matched_candidate['id'])
                                screenshot_saved_global = True

                    # --- LOGIC: SLEEP (Only if not texting) ---
                    if status != "texting" and person_crop.size > 0:
                        # Use person ID for sleep detector state key if possible, or fallback
                        sleep_key = f"{camera_name}_pid_{matched_candidate['id']}"
                        sleep_status, sleep_info = self.sleep_detector.process_crop(person_crop, id_key=sleep_key)
                        
                        if sleep_status == "sleeping":
                            matched_candidate['streak_sleep'] += 1
                            
                            # Check Sleep Threshold
                            if matched_candidate['streak_sleep'] >= self.SLEEP_CONSISTENCY_THRESHOLD:
                                status = "sleeping"
                                color = (255, 0, 0) # Blue (BGR)
                                if global_status != "texting":
                                    global_status = "sleeping"
                                display_text = f"P{matched_candidate['id']}: SLEEP"

                                # Check Cooldown
                                if save_screenshots:
                                    last_shot = matched_candidate['cooldowns']['sleep']
                                    if (current_time - last_shot) > 120:
                                        matched_candidate['cooldowns']['sleep'] = current_time
                                        self.save_evidence(frame, x1, y1, x2, y2, camera_name, "SLEEP", matched_candidate['id'])
                                        screenshot_saved_global = True
                        else:
                             matched_candidate['streak_sleep'] = max(0, matched_candidate['streak_sleep'] - 1)
                             if sleep_status == "drowsy":
                                 color = (0, 255, 255) # Yellow
                    else:
                        # If texting, reset sleep streak? Or keep it?
                        # Usually if texting, you aren't sleeping.
                        matched_candidate['streak_sleep'] = max(0, matched_candidate['streak_sleep'] - 1)

                    # Store for display
                    self.last_display_data.append((x1, y1, x2, y2, color, status, display_text))

            # --- PRUNING STREAKS ---
            # Remove streaks that were not matched in this frame (person left)
            self.detection_streaks = [
                d for d in self.detection_streaks 
                if (current_time - d['last_seen']) < 1.0 # 1 second tolerance
            ]

        # --- DRAWING (Every Frame using cached data) ---
        for (x1, y1, x2, y2, color, status, text) in self.last_display_data:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if text:
                 cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Global status update from cache if not updated in this frame (because skip_frames)
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
