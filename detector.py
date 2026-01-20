import cv2
import time
import os
import math
import datetime
from ultralytics import YOLO
import threading
from sleep_detector import SleepDetector

class PhoneDetector:
    def __init__(self, model_path='yolo26n.pt', output_dir="detections", cooldown_seconds=120, consistency_threshold=3, model_instance=None, pose_model_instance=None, lock=None):
        
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Handling Shared Model
        # Note: For tracking to work correctly per camera, model_instance should be None (private instance)
        self.lock = lock
        if model_instance:
            self.model = model_instance
        else:
            self.model = YOLO(model_path)
            
        # Initialize Sleep Detector
        self.sleep_detector = SleepDetector(pose_model_instance=pose_model_instance)

        self.PHONE_CLASS_ID = 67
        self.PERSON_CLASS_ID = 0
        self.COOLDOWN_SECONDS = cooldown_seconds # Default updated to 120 per requirements
        self.CONSISTENCY_THRESHOLD = consistency_threshold
        
        # Cooldown tracking: Dictionary {(track_id, type): last_screenshot_time}
        self.cooldowns = {}
        
        # Streak tracking: Dictionary {(track_id, type): current_streak_count}
        self.streaks = {}

        # State for frame skipping
        self.last_display_data = [] # List of (x1, y1, x2, y2, color, status, label_text)

    def process_frame(self, frame, frame_count, skip_frames=5, save_screenshots=True, conf_threshold=0.25, camera_name="Unknown"):
        """
        Process the frame.
        skip_frames: Run heavy inference only every N frames.
        Returns: frame, global_status_string, screenshot_saved_bool
        """
        current_time = time.time()
        
        # Cleanup old cooldowns (optional, to prevent memory leak over infinite time)
        if frame_count % 1000 == 0:
            self.cooldowns = {k: v for k, v in self.cooldowns.items() if (current_time - v) < self.COOLDOWN_SECONDS * 2}
            self.streaks = {} # Periodic cleanup of stale streaks

        global_status = "safe" # safe, texting, sleeping
        screenshot_saved_global = False

        # --- HEAVY INFERENCE STEP ---
        if frame_count % skip_frames == 0:
            self.last_display_data = [] # Reset display data
            
            # 1. First Pass: TRACK People
            # Thread-safe inference: logic simplified as we expect private model for tracking
            # but we keep lock check just in case legacy shared model is used
            
            classes_to_track = [self.PERSON_CLASS_ID]
            
            if self.lock:
                with self.lock:
                    results = self.model.track(frame, classes=classes_to_track, conf=conf_threshold, persist=True, verbose=False)
            else:
                results = self.model.track(frame, classes=classes_to_track, conf=conf_threshold, persist=True, verbose=False)
            
            if len(results) > 0 and results[0].boxes:
                boxes = results[0].boxes
                
                for box in boxes:
                    # Get Box Info
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Get Track ID
                    track_id = int(box.id.item()) if box.id is not None else None
                    
                    # Default status
                    status = "safe"
                    color = (0, 255, 0) # Green
                    label = f"ID: {track_id}" if track_id is not None else "Person"

                    # --- ZOOM LOGIC ---
                    h, w, _ = frame.shape
                    pad = 20
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(w, x2 + pad)
                    cy2 = min(h, y2 + pad)
                    
                    person_crop = frame[cy1:cy2, cx1:cx2]
                    
                    if person_crop.size > 0:
                        # 2. Second Pass: Run AI on JUST this person (Phone Detection)
                        # Predict is stateless, safe to use even if shared
                        
                        crop_results = self.model.predict(person_crop, classes=[self.PHONE_CLASS_ID], conf=0.15, verbose=False)
                        
                        phone_detected = False
                        if len(crop_results) > 0 and len(crop_results[0].boxes) > 0:
                            phone_detected = True
                            
                        if phone_detected:
                            status = "texting"
                            color = (0, 0, 255) # Red
                            global_status = "texting"

                        else:
                            # 3. Sleep Detection (If not texting)
                            # We use track_id as key for sleep state persistence if available
                            sleep_key = f"{camera_name}_id_{track_id}" if track_id is not None else f"{camera_name}_unknown"
                            sleep_status, _ = self.sleep_detector.process_crop(person_crop, id_key=sleep_key)
                            
                            if sleep_status == "sleeping":
                                status = "sleeping"
                                color = (255, 0, 0) # Blue (BGR)
                                if global_status != "texting":
                                    global_status = "sleeping"
                            elif sleep_status == "drowsy":
                                color = (0, 255, 255) # Yellow

                        # --- VERIFICATION STREAK ---
                        # We need to confirm the violation persists for N frames to avoid glitches

                        violation_type = None
                        if status == "texting": violation_type = "texting"
                        elif status == "sleeping": violation_type = "sleeping"

                        current_streak_val = 0

                        if track_id is not None:
                            # 1. Reset streaks for OTHER types for this ID (e.g. if now texting, reset sleep streak)
                            # Actually, simpler: Just track the current active violation.

                            streak_key = (track_id, violation_type)

                            # Increment Streak
                            if violation_type:
                                self.streaks[streak_key] = self.streaks.get(streak_key, 0) + 1
                                current_streak_val = self.streaks[streak_key]

                            # Reset streaks for non-active violations
                            # If we are currently "texting", we should reset the "sleeping" streak for this person to 0
                            # If we are "safe" (violation_type is None), we reset ALL streaks for this person.

                            possible_types = ["texting", "sleeping"]
                            for v_type in possible_types:
                                if v_type != violation_type:
                                     self.streaks[(track_id, v_type)] = 0

                            # --- COOLDOWN & SAVING ---
                            # Only save if:
                            # 1. We have a valid Track ID
                            # 2. We have a violation
                            # 3. Streak >= Threshold (Verification)
                            # 4. Cooldown expired

                            if save_screenshots and violation_type:
                                if current_streak_val >= self.CONSISTENCY_THRESHOLD:

                                    key = (track_id, violation_type)
                                    last_time = self.cooldowns.get(key, 0)

                                    if (current_time - last_time) > self.COOLDOWN_SECONDS:
                                        # Save Evidence
                                        type_str = "PHONE" if violation_type == "texting" else "SLEEP"
                                        self.save_evidence(frame, x1, y1, x2, y2, camera_name, type_str, track_id=track_id)
                                        screenshot_saved_global = True

                                        # Update Cooldown
                                        self.cooldowns[key] = current_time

                                        # Visual feedback
                                        label += " [SAVED]"
                                    else:
                                        # Cooldown active
                                        pass
                                else:
                                    # Building streak
                                    label += f" [{current_streak_val}/{self.CONSISTENCY_THRESHOLD}]"

                    # Store for display
                    self.last_display_data.append((x1, y1, x2, y2, color, status, label))

        # --- DRAWING (Every Frame using cached data) ---
        for (x1, y1, x2, y2, color, status, label) in self.last_display_data:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw Label
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw Status if meaningful
            if status != "safe":
                status_text = "PHONE" if status == "texting" else status.upper()
                cv2.putText(frame, status_text, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if status == "texting":
                global_status = "texting"
            elif status == "sleeping" and global_status != "texting":
                global_status = "sleeping"

        return frame, global_status, screenshot_saved_global

    def save_evidence(self, frame, x1, y1, x2, y2, camera_name="Unknown", detection_type="PHONE", track_id=None):
        """
        Save evidence screenshot.
        
        Args:
            detection_type: "PHONE" or "SLEEP"
            track_id: ID of the person (optional)
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
        header_text = f"{detection_type} | {camera_name} | {ts}"
        if track_id is not None:
             header_text += f" | ID: {track_id}"

        cv2.putText(evidence_img, header_text, (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        timestamp_fn = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3] # Add ms
        # Sanitize camera name for filename
        safe_cam_name = "".join([c for c in camera_name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        
        # Add detection type and ID to filename to ensure uniqueness
        id_str = f"_id{track_id}" if track_id is not None else ""
        filename = os.path.join(self.output_dir, f"evidence_{detection_type.lower()}_{safe_cam_name}_{timestamp_fn}{id_str}.jpg")
        cv2.imwrite(filename, evidence_img)
        print(f"📸 EVIDENCE SAVED: {filename}")
