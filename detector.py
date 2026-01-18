import cv2
import time
import os
import math
import datetime
from ultralytics import YOLO
import threading

class PhoneDetector:
    def __init__(self, model_path='yolo11s.pt', output_dir="detections", cooldown_seconds=5, consistency_threshold=3, model_instance=None, lock=None):
        
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Handling Shared Model
        self.lock = lock
        if model_instance:
            self.model = model_instance
        else:
            self.model = YOLO(model_path)
            
        self.PHONE_CLASS_ID = 67
        self.PERSON_CLASS_ID = 0
        self.COOLDOWN_SECONDS = cooldown_seconds
        
        # Temporal consistency: list of {'center': (x,y), 'streak': int, 'last_seen': time}
        self.detection_streaks = []
        self.CONSISTENCY_THRESHOLD = consistency_threshold
        
        # Spatial cooldown tracking: list of (x, y, timestamp)
        self.screenshot_locations = []
        
        # State for frame skipping
        self.last_display_data = [] # List of (x1, y1, x2, y2, color, status)

    def process_frame(self, frame, frame_count, skip_frames=5, save_screenshots=True, conf_threshold=0.25, camera_name="Unknown"):
        """
        Process the frame.
        skip_frames: Run heavy inference only every N frames.
        """
        current_time = time.time()
        
        # Cleanup old cooldowns (older than cooldown_seconds)
        self.screenshot_locations = [
            s for s in self.screenshot_locations 
            if (current_time - s[2]) < self.COOLDOWN_SECONDS
        ]

        phone_detected_global = False
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
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    p_cx, p_cy = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # Default status
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
                    
                    is_candidate = False
                    
                    if person_crop.size > 0:
                        # 2. Second Pass: Run AI on JUST this person
                        # Look for PHONE class (67) with lower threshold
                        
                        # Thread-safe inference
                        if self.lock:
                            with self.lock:
                                crop_results = self.model.predict(person_crop, classes=[self.PHONE_CLASS_ID], conf=0.15, verbose=False)
                        else:
                            crop_results = self.model.predict(person_crop, classes=[self.PHONE_CLASS_ID], conf=0.15, verbose=False)
                        
                        if len(crop_results) > 0 and len(crop_results[0].boxes) > 0:
                            is_candidate = True
                    
                    if is_candidate:
                        # --- TEMPORAL CONSISTENCY ---
                        # Try to match with existing streak
                        matched = False
                        for candidate in self.detection_streaks:
                            lx, ly = candidate['center']
                            dist = math.sqrt((p_cx - lx)**2 + (p_cy - ly)**2)
                            
                            # Match if within 100 pixels
                            if dist < 100: 
                                candidate['streak'] += 1
                                candidate['center'] = (p_cx, p_cy)
                                candidate['last_seen'] = current_time
                                matched = True
                                
                                # Check Threshold
                                if candidate['streak'] >= self.CONSISTENCY_THRESHOLD:
                                    status = "texting"
                                    color = (0, 0, 255) # Red
                                    phone_detected_global = True
                                    
                                    # Attempt Save
                                    if save_screenshots:
                                        # Check spatial cooldown
                                        should_save = True
                                        for (sx, sy, stime) in self.screenshot_locations:
                                            if math.sqrt((p_cx - sx)**2 + (p_cy - sy)**2) < 100:
                                                should_save = False; break
                                        
                                        if should_save:
                                            self.screenshot_locations.append((p_cx, p_cy, current_time))
                                            self.save_evidence(frame, x1, y1, x2, y2, camera_name)
                                            screenshot_saved_global = True
                                break
                        
                        if not matched:
                            # New Candidate
                            self.detection_streaks.append({
                                'center': (p_cx, p_cy),
                                'streak': 1,
                                'last_seen': current_time
                            })
                            # Keep status green until threshold reached
                            
                        # Mark this person as processed (so we don't prune their streak)
                        current_frame_detections.append((p_cx, p_cy))
                            
                    # Store for display
                    self.last_display_data.append((x1, y1, x2, y2, color, status))

            # --- PRUNING STREAKS ---
            # Remove streaks that were not matched in this frame (person stopped using phone or left)
            # We match if the streak's last_seen is very close to current_time
            # Since we just updated 'last_seen' for matched candidates, we can use that.
            self.detection_streaks = [
                d for d in self.detection_streaks 
                if (current_time - d['last_seen']) < 1.0 # 1 second tolerance
            ]

        # --- DRAWING (Every Frame using cached data) ---
        for (x1, y1, x2, y2, color, status) in self.last_display_data:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if status == "texting":
                cv2.putText(frame, "PHONE DETECTED", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Keep the global flag true if we are still displaying the alert
                phone_detected_global = True 

        return frame, phone_detected_global, screenshot_saved_global

    def save_evidence(self, frame, x1, y1, x2, y2, camera_name="Unknown"):
        evidence_img = frame.copy()
        
        # Draw the box on the evidence
        cv2.rectangle(evidence_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Add Header
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Draw black bar at top
        cv2.rectangle(evidence_img, (0, 0), (evidence_img.shape[1], 40), (0,0,0), -1)
        # Add text
        header_text = f"PHONE | {camera_name} | {ts}"
        cv2.putText(evidence_img, header_text, (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        timestamp_fn = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Sanitize camera name for filename
        safe_cam_name = "".join([c for c in camera_name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        
        filename = os.path.join(self.output_dir, f"evidence_{safe_cam_name}_{timestamp_fn}.jpg")
        cv2.imwrite(filename, evidence_img)
        print(f"📸 EVIDENCE SAVED: {filename}")
