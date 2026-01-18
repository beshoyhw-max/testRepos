import cv2
import threading
import time
import json
import os
import datetime
from detector import PhoneDetector
from ultralytics import YOLO

# Shared Lock for Model Inference to prevent race conditions/OOM if using GPU
model_lock = threading.Lock()

class CameraThread(threading.Thread):
    def __init__(self, camera_config, shared_model, shared_pose_model, conf_threshold=0.25):
        super().__init__()
        self.camera_id = camera_config['id']
        self.camera_name = camera_config['name']
        self.source = camera_config['source']
        self.shared_model = shared_model
        self.shared_pose_model = shared_pose_model
        
        # Initialize independent detector state for this camera
        # We pass the shared model to it (requires update in detector.py)
        self.detector = PhoneDetector(
            model_instance=self.shared_model,
            pose_model_instance=self.shared_pose_model,
            lock=model_lock
        )
        
        self.conf_threshold = conf_threshold
        self.running = False
        self.latest_frame = None
        self.status = "safe"
        self.is_connected = False
        self.last_update_time = 0
        
    def run(self):
        self.running = True
        print(f"[{self.camera_name}] Starting thread...")
        
        while self.running:
            # Reconnect loop
            cap = cv2.VideoCapture(self.source)
            # Try to optimize resolution if it's a webcam (int source)
            if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
            if not cap.isOpened():
                print(f"[{self.camera_name}] Failed to open source. Retrying in 5s...")
                self.is_connected = False
                self.status = "error"
                time.sleep(5)
                continue
                
            self.is_connected = True
            print(f"[{self.camera_name}] Connected.")
            
            frame_count = 0
            # Enterprise Governor: Sleep slightly to allow other threads to run
            # We can tune this dynamic sleep later.
            
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print(f"[{self.camera_name}] Stream lost.")
                    break
                
                # Run Detection
                # We handle the frame skipping inside the thread loop or inside the detector.
                # Let's do it here to save locking overhead.
                SKIP_FRAMES = 3
                
                try:
                    # process_frame now returns (frame, status_string, is_saved)
                    processed_frame, status, is_saved = self.detector.process_frame(
                        frame, 
                        frame_count, 
                        skip_frames=SKIP_FRAMES, 
                        save_screenshots=True, # We can make this configurable
                        conf_threshold=self.conf_threshold,
                        camera_name=self.camera_name # For file naming
                    )
                    
                    self.latest_frame = processed_frame
                    self.status = status
                    self.last_update_time = time.time()
                    
                except Exception as e:
                    print(f"[{self.camera_name}] Error in processing: {e}")
                
                frame_count += 1
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)
                
            cap.release()
            self.is_connected = False
            
        print(f"[{self.camera_name}] Thread stopped.")

    def stop(self):
        self.running = False
        self.join()

    def get_frame(self):
        return self.latest_frame

    def get_status(self):
        # If data is stale (> 3 seconds), consider it disconnected
        if time.time() - self.last_update_time > 3.0:
            return "disconnected"
        return self.status

class CameraManager:
    def __init__(self, config_file="cameras.json"):
        self.config_file = config_file
        self.cameras = {} # id -> CameraThread
        self.shared_model = None
        self.shared_pose_model = None
        
        # Load Model Once
        print("Loading Shared YOLO Model (Detection)...")
        self.shared_model = YOLO('yolo11n.pt')
        print("Loading Shared YOLO Model (Pose)...")
        self.shared_pose_model = YOLO('yolo11n-pose.pt')
        print("Models Loaded.")
        
        self.load_config_and_start()

    def load_config_and_start(self):
        if not os.path.exists(self.config_file):
            # Default config: Webcam
            default_config = [
                {"id": 0, "name": "Webcam Main", "source": 0}
            ]
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f)
        
        with open(self.config_file, 'r') as f:
            configs = json.load(f)
            
        # Start threads for each config
        for conf in configs:
            self.add_camera_thread(conf)

    def add_camera_thread(self, config):
        # Convert source to int if it's a digit (for webcam index)
        source = config['source']
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        config['source'] = source

        cam_id = config['id']
        if cam_id in self.cameras:
            return # Already running
            
        thread = CameraThread(config, self.shared_model, self.shared_pose_model)
        thread.start()
        self.cameras[cam_id] = thread

    def add_camera(self, name, source):
        # Generate new ID
        existing_ids = [c.camera_id for c in self.cameras.values()]
        new_id = max(existing_ids) + 1 if existing_ids else 0
        
        new_config = {"id": new_id, "name": name, "source": source}
        
        # Update JSON
        self.save_config_append(new_config)
        
        # Start Thread
        self.add_camera_thread(new_config)

    def remove_camera(self, cam_id):
        if cam_id in self.cameras:
            print(f"Removing camera {cam_id}...")
            self.cameras[cam_id].stop()
            del self.cameras[cam_id]
            
            # Update JSON
            self.save_config_remove(cam_id)

    def save_config_append(self, new_config):
        with open(self.config_file, 'r') as f:
            configs = json.load(f)
        configs.append(new_config)
        with open(self.config_file, 'w') as f:
            json.dump(configs, f)

    def save_config_remove(self, cam_id):
        with open(self.config_file, 'r') as f:
            configs = json.load(f)
        configs = [c for c in configs if c['id'] != cam_id]
        with open(self.config_file, 'w') as f:
            json.dump(configs, f)

    def get_active_cameras(self):
        return self.cameras

    def update_global_conf(self, conf):
        for cam in self.cameras.values():
            cam.conf_threshold = conf
