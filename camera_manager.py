import cv2
import threading
import time
import json
import os
import datetime
from detector import PhoneDetector
from ultralytics import YOLO

# Force TCP connection (critical for Huawei cameras and general RTSP stability)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Shared Lock for Model Inference to prevent race conditions/OOM if using GPU
model_lock = threading.Lock()

class VideoReader:
    """
    Dedicated thread for reading frames from a video source.
    Ensures that we always have the latest frame available, discarding older ones.
    Solves the producer-consumer lag issue.
    """
    def __init__(self, source, camera_name="Unknown"):
        self.source = source
        self.camera_name = camera_name
        
        self.cap = None
        self.frame = None
        self.last_read_time = 0
        self.running = False
        self.connected = False
        
        # Start reading thread
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        
    def start(self):
        self.running = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def update(self):
        print(f"[{self.camera_name}] VideoReader started for source: {self.source}")
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.connected = False
                print(f"[{self.camera_name}] Connecting to source...")
                self.cap = cv2.VideoCapture(self.source)
                
                # Optimize for webcam
                if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # Critical for low latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                
                if not self.cap.isOpened():
                    print(f"[{self.camera_name}] Connection failed. Retrying in 5s...")
                    time.sleep(5)
                    continue
                
                print(f"[{self.camera_name}] Connected.")
                self.connected = True

            # Read frame
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                    self.last_read_time = time.time()
                else:
                    # Stream lost or end of file
                    print(f"[{self.camera_name}] Stream read failed.")
                    self.cap.release()
                    self.connected = False
                    time.sleep(0.5) # Wait before retry
            except Exception as e:
                print(f"[{self.camera_name}] Error reading frame: {e}")
                self.connected = False
                if self.cap:
                    self.cap.release()
                time.sleep(1)

    def get_frame(self):
        return self.frame, self.last_read_time
        
    def is_connected(self):
        # Consider connected if we read a frame recently
        return self.connected and (time.time() - self.last_read_time < 3.0)


class CameraThread(threading.Thread):
    def __init__(self, camera_config, shared_model, shared_pose_model, conf_threshold=0.25):
        super().__init__()
        self.camera_id = camera_config['id']
        self.camera_name = camera_config['name']
        self.source = camera_config['source']
        self.shared_model = shared_model
        self.shared_pose_model = shared_pose_model
        
        # Initialize independent detector state for this camera
        # Pass model_instance=None to force loading a fresh private model for tracking
        self.detector = PhoneDetector(
            model_instance=None,
            pose_model_instance=self.shared_pose_model,
            lock=None # No lock needed for private model
        )
        
        self.conf_threshold = conf_threshold
        self.running = False
        self.latest_processed_frame = None
        self.status = "safe"
        self.last_update_time = 0
        self.last_processed_timestamp = 0
        
        # Initialize VideoReader
        self.reader = VideoReader(self.source, self.camera_name)
        
    def run(self):
        self.running = True
        print(f"[{self.camera_name}] Starting processing thread...")
        self.reader.start()
        
        frame_count = 0
        
        while self.running:
            # Check connection status
            if not self.reader.is_connected():
                self.status = "disconnected"
                time.sleep(0.5)
                continue
            
            # Get latest frame from reader
            raw_frame, timestamp = self.reader.get_frame()
            
            # Skip if no frame or if we already processed this frame
            if raw_frame is None or timestamp == self.last_processed_timestamp:
                # Frame not ready or duplicate
                time.sleep(0.01)
                continue
            
            self.last_processed_timestamp = timestamp

            # Process Frame
            # process_frame returns (frame, status_string, is_saved)
            # We process EVERY frame we get from the reader (which is already skipping frames naturally)
            # But we still pass frame_count to detector for its internal consistency checks (skip_frames arg)
            
            try:
                processed_frame, status, is_saved = self.detector.process_frame(
                    raw_frame, 
                    frame_count, 
                    skip_frames=3, # Still skip internally if needed for performance
                    save_screenshots=True,
                    conf_threshold=self.conf_threshold,
                    camera_name=self.camera_name
                )
                
                self.latest_processed_frame = processed_frame
                self.status = status
                self.last_update_time = time.time()
                
            except Exception as e:
                print(f"[{self.camera_name}] Error in processing: {e}")
            
            frame_count += 1
            
            # Small sleep to prevent CPU hogging in this loop
            time.sleep(0.01)

        print(f"[{self.camera_name}] Processing thread stopped.")
        self.reader.stop()

    def stop(self):
        self.running = False
        self.join()

    def get_frame(self):
        return self.latest_processed_frame

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
        # Note: Detection model is now loaded per-camera to support tracking persistence
        print("Loading Shared YOLO Model (Pose)...")
        self.shared_pose_model = YOLO('yolo26n-pose.pt')
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
            
        thread = CameraThread(config, None, self.shared_pose_model)
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
