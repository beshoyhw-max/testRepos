import cv2
import time
import os
import math
import datetime
from ultralytics import YOLO
import threading
from sleep_detector import SleepDetector

class PhoneDetector:
    def __init__(self, model_path='yolo11s.pt', output_dir="detections", cooldown_seconds=5, consistency_threshold=3, model_instance=None, pose_model_instance=None, lock=None):
        
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
        
        # Temporal consistency: list of {'center': (x,y), 'streak': int, 'last_seen': time}
        self.detection_streaks = []
        self.CONSISTENCY_THRESHOLD = consistency_threshold
        
        # Sleep Detection Tracking (Similar to phone detection)
        self.sleep_streaks = []  # Track sleep consistency per person
        self.SLEEP_CONSISTENCY_THRESHOLD = 3  # Need 3 consecutive detections
        
        # Spatial cooldown tracking: list of (x, y, timestamp, type)
        # type can be 'phone' or 'sleep'
        self.screenshot_locations = []
        
        # State for frame skipping
        self.last_display_data = [] # List of (x1, y1, x2, y2, color, status)

        # Async Saving Queue
        self.save_queue = []
        self.save_thread_lock = threading.Lock()

    def _save_evidence_async_worker(self, evidence_img, filename):
        """Worker to save image in background"""
        try:
            cv2.imwrite(filename, evidence_img)
            print(f"📸 EVIDENCE SAVED: {filename}")
        except Exception as e:
            print(f"Error saving evidence: {e}")

    def process_frame(self, frame, frame_count, skip_frames=5, save_screenshots=True, conf_threshold=0.25, camera_name="Unknown"):
        """
        Process the frame.
        skip_frames: Run heavy inference only every N frames.
        Returns: frame, global_status_string, screenshot_saved_bool
        """
        current_time = time.time()
        
        # Cleanup old cooldowns (older than cooldown_seconds)
        # Format: (x, y, timestamp, type)
        self.screenshot_locations = [
            s for s in self.screenshot_locations 
            if (current_time - s[2]) < self.COOLDOWN_SECONDS
        ]

        global_status = "safe" # safe, texting, sleeping
        screenshot_saved_global = False

        # --- HEAVY INFERENCE STEP ---
        if frame_count % skip_frames == 0:
            self.last_display_data = [] # Reset display data
            
            # Temporary list to track who we saw this frame (for streak management)
            current_frame_detections = [] 
            
            # 1. First Pass: Find People in the full room
            # We filter for PERSON class (0) only
            
            # OPTIMIZATION: Resize frame for inference if it's huge (e.g. 4k)
            # YOLO works best on 640. Standard webcams are 720p or 1080p.
            # We pass the full frame but let YOLO handle resizing internally usually.
            # However, explictly ensuring we don't pass massive arrays can help.
            # For now, we trust Ultralytics auto-resize but we ensure half-precision if supported.

            # Thread-safe inference
            # We use `half=True` (FP16) if possible, but it requires CUDA. CPU runs FP32.
            # Ultralytics handles this automatically if device=0.

            if self.lock:
                with self.lock:
                    results = self.model.predict(frame, classes=[self.PERSON_CLASS_ID], conf=conf_threshold, verbose=False, imgsz=640)
            else:
                results = self.model.predict(frame, classes=[self.PERSON_CLASS_ID], conf=conf_threshold, verbose=False, imgsz=640)
            
            # Use cpu().numpy() or .tolist() to get coordinates
            if len(results) > 0:
                boxes = results[0].boxes
                
                for idx, box in enumerate(boxes):
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
                                    global_status = "texting"
                                    
                                    # Attempt Save
                                    if save_screenshots:
                                        # Check spatial cooldown
                                        should_save = True
                                        for (sx, sy, stime, stype) in self.screenshot_locations:
                                            if stype == 'phone' and math.sqrt((p_cx - sx)**2 + (p_cy - sy)**2) < 100:
                                                should_save = False
                                                break
                                        
                                        if should_save:
                                            self.screenshot_locations.append((p_cx, p_cy, current_time, 'phone'))
                                            self.save_evidence(frame, x1, y1, x2, y2, camera_name, "PHONE")
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

                    # --- SLEEP DETECTION (If not texting) ---
                    if status != "texting" and person_crop.size > 0:
                        # We use a simple ID key based on camera + index to persist state roughly
                        # Ideally we would use tracking ID, but we don't have it yet.
                        # Assumption: People don't swap seats often.
                        sleep_key = f"{camera_name}_idx_{idx}"
                        sleep_status, sleep_info = self.sleep_detector.process_crop(person_crop, id_key=sleep_key)
                        

                        if sleep_status == "sleeping":
                            # ADD TEMPORAL CONSISTENCY FOR SLEEP
                            # Similar logic to phone detection
                            matched_sleep = False
                            for sleep_candidate in self.sleep_streaks:
                                sx, sy = sleep_candidate['center']
                                dist = math.sqrt((p_cx - sx)**2 + (p_cy - sy)**2)
                                
                                if dist < 100:
                                    sleep_candidate['streak'] += 1
                                    sleep_candidate['center'] = (p_cx, p_cy)
                                    sleep_candidate['last_seen'] = current_time
                                    matched_sleep = True
                                    
                                    # Check if we should mark as sleeping and save
                                    if sleep_candidate['streak'] >= self.SLEEP_CONSISTENCY_THRESHOLD:
                                        status = "sleeping"
                                        color = (255, 0, 0) # Blue (BGR)
                                        if global_status != "texting":
                                            global_status = "sleeping"
                                        
                                        # SAVE SLEEP SCREENSHOT
                                        if save_screenshots:
                                            should_save = True
                                            for (sx2, sy2, stime, stype) in self.screenshot_locations:
                                                if stype == 'sleep' and math.sqrt((p_cx - sx2)**2 + (p_cy - sy2)**2) < 100:
                                                    should_save = False
                                                    break
                                            
                                            if should_save:
                                                self.screenshot_locations.append((p_cx, p_cy, current_time, 'sleep'))
                                                self.save_evidence(frame, x1, y1, x2, y2, camera_name, "SLEEP")
                                                screenshot_saved_global = True
                                    break
                            
                            if not matched_sleep:
                                # New sleep candidate
                                self.sleep_streaks.append({
                                    'center': (p_cx, p_cy),
                                    'streak': 1,
                                    'last_seen': current_time
                                })
                            
                        elif sleep_status == "drowsy":
                            # Warning color
                            color = (0, 255, 255) # Yellow
                            
                    # Store for display
                    self.last_display_data.append((x1, y1, x2, y2, color, status))

            # --- PRUNING STREAKS ---
            # Remove streaks that were not matched in this frame (person stopped using phone or left)
            self.detection_streaks = [
                d for d in self.detection_streaks 
                if (current_time - d['last_seen']) < 1.0 # 1 second tolerance
            ]
            
            # PRUNE SLEEP STREAKS
            self.sleep_streaks = [
                d for d in self.sleep_streaks
                if (current_time - d['last_seen']) < 1.0
            ]

        # --- DRAWING (Every Frame using cached data) ---
        for (x1, y1, x2, y2, color, status) in self.last_display_data:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if status == "texting":
                cv2.putText(frame, "PHONE DETECTED", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                global_status = "texting" # Ensure drawing updates global status if cached
            elif status == "sleeping":
                cv2.putText(frame, "SLEEPING", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if global_status != "texting":
                     global_status = "sleeping"

        return frame, global_status, screenshot_saved_global

    def save_evidence(self, frame, x1, y1, x2, y2, camera_name="Unknown", detection_type="PHONE"):
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
        header_text = f"{detection_type} | {camera_name} | {ts}"
        cv2.putText(evidence_img, header_text, (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        timestamp_fn = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Sanitize camera name for filename
        safe_cam_name = "".join([c for c in camera_name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        
        # Add detection type to filename
        filename = os.path.join(self.output_dir, f"evidence_{detection_type.lower()}_{safe_cam_name}_{timestamp_fn}.jpg")

        # Launch background thread for saving
        threading.Thread(target=self._save_evidence_async_worker, args=(evidence_img, filename), daemon=True).start()