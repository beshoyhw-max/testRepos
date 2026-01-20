import cv2
import time
import os
import math
import datetime
from ultralytics import YOLO
import threading
from sleep_detector import SleepDetector

class PhoneDetector:
    def __init__(self, model_path='yolo11s.pt', output_dir="detections", cooldown_seconds=120, consistency_threshold=3, model_instance=None, pose_model_instance=None, lock=None):
        
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Handling Independent Model
        # We ignore lock for self.model because it is now per-thread (independent)
        if model_instance:
            self.model = model_instance
        else:
            self.model = YOLO(model_path)
            
        # Initialize Sleep Detector
        self.sleep_detector = SleepDetector(pose_model_instance=pose_model_instance)

        self.PHONE_CLASS_ID = 67
        self.PERSON_CLASS_ID = 0
        self.COOLDOWN_SECONDS = cooldown_seconds # 120s per person
        
        # ID-based Tracking History
        # Key: track_id (int)
        # Value: {'phone_streak': int, 'sleep_streak': int, 'last_seen': float}
        self.track_history = {}
        self.CONSISTENCY_THRESHOLD = consistency_threshold
        self.SLEEP_CONSISTENCY_THRESHOLD = 3
        
        # ID-based Violation Cooldowns
        # Key: (track_id, violation_type)  e.g., (5, 'PHONE')
        # Value: timestamp (float)
        self.violation_cooldowns = {}
        
        # State for frame skipping (Visual Cache)
        self.last_display_data = [] # List of (x1, y1, x2, y2, color, status, label_text)

    def process_frame(self, frame, frame_count, skip_frames=5, save_screenshots=True, conf_threshold=0.25, camera_name="Unknown"):
        """
        Process the frame using YOLO Tracking.
        skip_frames: Run heavy inference only every N frames.
        Returns: frame, global_status_string, screenshot_saved_bool
        """
        current_time = time.time()
        
        # Cleanup old cooldowns to prevent memory leaks
        # Remove entries older than 2 * COOLDOWN_SECONDS (safe margin)
        keys_to_remove = [k for k, v in self.violation_cooldowns.items() if (current_time - v) > (self.COOLDOWN_SECONDS * 2)]
        for k in keys_to_remove:
            del self.violation_cooldowns[k]

        # Cleanup old track history (people who left)
        # Remove entries not seen for > 10 seconds
        ids_to_remove = [k for k, v in self.track_history.items() if (current_time - v['last_seen']) > 10.0]
        for k in ids_to_remove:
            del self.track_history[k]

        global_status = "safe" # safe, texting, sleeping
        screenshot_saved_global = False

        # --- HEAVY INFERENCE STEP ---
        if frame_count % skip_frames == 0:
            self.last_display_data = [] # Reset display data
            
            # 1. Track People
            # persist=True enables tracking (ID assignment)
            results = self.model.track(frame, classes=[self.PERSON_CLASS_ID], conf=conf_threshold, persist=True, verbose=False)
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for idx, box in enumerate(boxes):
                    # Extract ID
                    track_id = int(box.id.item()) if box.id is not None else -1

                    # Coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Initialize State for new ID
                    if track_id != -1:
                        if track_id not in self.track_history:
                            self.track_history[track_id] = {
                                'phone_streak': 0,
                                'sleep_streak': 0,
                                'last_seen': current_time
                            }
                        self.track_history[track_id]['last_seen'] = current_time

                    # Default status for this person
                    status = "safe"
                    color = (0, 255, 0) # Green
                    
                    # --- ZOOM LOGIC ---
                    h, w, _ = frame.shape
                    pad = 20
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(w, x2 + pad)
                    cy2 = min(h, y2 + pad)
                    
                    person_crop = frame[cy1:cy2, cx1:cx2]
                    
                    # === CHECK 1: PHONE DETECTION ===
                    is_phone_detected = False
                    if person_crop.size > 0:
                        # Run AI on JUST this person crop
                        # Note: We use predict() here, not track(), so it's stateless on the crop
                        crop_results = self.model.predict(person_crop, classes=[self.PHONE_CLASS_ID], conf=0.15, verbose=False)
                        if len(crop_results) > 0 and len(crop_results[0].boxes) > 0:
                            is_phone_detected = True
                    
                    if is_phone_detected and track_id != -1:
                        self.track_history[track_id]['phone_streak'] += 1
                        
                        # Check Threshold
                        if self.track_history[track_id]['phone_streak'] >= self.CONSISTENCY_THRESHOLD:
                            status = "texting"
                            color = (0, 0, 255) # Red
                            global_status = "texting"
                            
                            # Check Cooldown
                            last_shot = self.violation_cooldowns.get((track_id, "PHONE"), 0)
                            if save_screenshots and (current_time - last_shot) > self.COOLDOWN_SECONDS:
                                self.violation_cooldowns[(track_id, "PHONE")] = current_time
                                self.save_evidence(frame, x1, y1, x2, y2, camera_name, "PHONE", track_id)
                                screenshot_saved_global = True
                    elif track_id != -1:
                        # Reset streak if not detected
                        self.track_history[track_id]['phone_streak'] = max(0, self.track_history[track_id]['phone_streak'] - 1)

                    # === CHECK 2: SLEEP DETECTION (If not texting) ===
                    if status != "texting" and person_crop.size > 0 and track_id != -1:
                        # Use track_id for robust sleep state tracking
                        sleep_key = f"{camera_name}_id_{track_id}"
                        sleep_status, sleep_info = self.sleep_detector.process_crop(person_crop, id_key=sleep_key)
                        
                        if sleep_status == "sleeping":
                            self.track_history[track_id]['sleep_streak'] += 1
                            
                            if self.track_history[track_id]['sleep_streak'] >= self.SLEEP_CONSISTENCY_THRESHOLD:
                                status = "sleeping"
                                color = (255, 0, 0) # Blue (BGR) -> wait, text usually white, box blue
                                if global_status != "texting":
                                    global_status = "sleeping"

                                # Check Cooldown
                                last_shot = self.violation_cooldowns.get((track_id, "SLEEP"), 0)
                                if save_screenshots and (current_time - last_shot) > self.COOLDOWN_SECONDS:
                                    self.violation_cooldowns[(track_id, "SLEEP")] = current_time
                                    self.save_evidence(frame, x1, y1, x2, y2, camera_name, "SLEEP", track_id)
                                    screenshot_saved_global = True
                        elif sleep_status == "drowsy":
                             color = (0, 255, 255) # Yellow
                             # Do not reset streak immediately, or handle differently?
                             # For now, if drowsy, we don't increment sleep streak but don't reset hard?
                             # Let's keep it simple: reset if not "sleeping".
                             # Actually, if drowsy, maybe we shouldn't reset.
                             pass
                        else:
                            # Awake
                            self.track_history[track_id]['sleep_streak'] = 0

                    # Store for display
                    self.last_display_data.append((x1, y1, x2, y2, color, status, track_id))

        # --- DRAWING (Every Frame using cached data) ---
        for (x1, y1, x2, y2, color, status, tid) in self.last_display_data:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = ""
            if status == "texting":
                label = "PHONE"
            elif status == "sleeping":
                label = "SLEEPING"

            # Show ID for debugging/verification if available
            id_text = f"ID: {tid}" if tid != -1 else ""
            full_text = f"{label} {id_text}".strip()

            if full_text:
                cv2.putText(frame, full_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Update global status for return if drawing from cache
            if status == "texting":
                global_status = "texting"
            elif status == "sleeping" and global_status != "texting":
                global_status = "sleeping"

        return frame, global_status, screenshot_saved_global

    def save_evidence(self, frame, x1, y1, x2, y2, camera_name="Unknown", detection_type="PHONE", track_id=-1):
        """
        Save evidence screenshot.
        
        Args:
            detection_type: "PHONE" or "SLEEP"
        """
        evidence_img = frame.copy()
        
        # Draw the box on the evidence (High visibility)
        box_color = (0, 0, 255) if detection_type == "PHONE" else (255, 0, 0)  # Red for phone, Blue for sleep
        cv2.rectangle(evidence_img, (x1, y1), (x2, y2), box_color, 4)
        
        # Add Header
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Draw black bar at top
        cv2.rectangle(evidence_img, (0, 0), (evidence_img.shape[1], 40), (0,0,0), -1)
        # Add text
        header_text = f"{detection_type} | ID: {track_id} | {camera_name} | {ts}"
        cv2.putText(evidence_img, header_text, (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        timestamp_fn = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Sanitize camera name for filename
        safe_cam_name = "".join([c for c in camera_name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        
        # Add detection type and ID to filename
        filename = os.path.join(self.output_dir, f"evidence_{detection_type.lower()}_{safe_cam_name}_id{track_id}_{timestamp_fn}.jpg")
        cv2.imwrite(filename, evidence_img)
        print(f"📸 EVIDENCE SAVED: {filename}")
