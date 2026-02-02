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

                # FIXED: Clean up old capture before creating new one
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None

                self.cap = cv2.VideoCapture(self.source)
                
                # Optimize for webcam
                is_webcam = isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit())
                
                if is_webcam:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # Critical for low latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                
                # Check for file source to throttle FPS
                self.fps = 30 # Default
                self.is_file = False
                if not is_webcam and isinstance(self.source, str) and not self.source.startswith("rtsp"):
                     # Assume file if not int and not rtsp
                     if os.path.exists(self.source):
                         self.is_file = True
                         self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                         if self.fps <= 0:
                             self.fps = 30
                         print(f"[{self.camera_name}] File detected. FPS: {self.fps}")

                if not self.cap.isOpened():
                    print(f"[{self.camera_name}] Connection failed. Retrying in 5s...")
                    # FIXED: Clean up failed connection attempt
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                    time.sleep(5)
                    continue
                
                print(f"[{self.camera_name}] Connected successfully.")
                self.connected = True

            # Read frame
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                    self.last_read_time = time.time()
                    
                    # Throttle if file
                    if self.is_file:
                        time.sleep(1.0 / self.fps)
                        
                else:
                    # Stream lost or end of file
                    print(f"[{self.camera_name}] Stream read failed.")
                    # FIXED: Clean up before retry
                    if self.cap:
                        try:
                            self.cap.release()
                        except:
                            pass
                        self.cap = None
                    self.connected = False
                    time.sleep(0.5)

            except Exception as e:
                print(f"[{self.camera_name}] Error reading frame: {e}")
                self.connected = False
                # FIXED: Clean up on error
                if self.cap:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                time.sleep(1)

    def get_frame(self):
        return self.frame, self.last_read_time
        
    def is_connected(self):
        # Consider connected if we read a frame recently
        return self.connected and (time.time() - self.last_read_time < 3.0)


class CameraThread(threading.Thread):
    def __init__(self, camera_config, conf_threshold=0.25):
        """
        FIXED: No shared models - each camera gets private instances.
        This ensures complete thread safety and tracking persistence.
        """
        super().__init__()
        self.camera_id = camera_config['id']
        self.camera_name = camera_config['name']
        self.source = camera_config['source']
        
        # FIXED: Each camera loads its own private models
        # No more shared models = no race conditions!
        print(f"[{self.camera_name}] Initializing private detector...")
        self.detector = PhoneDetector(
            model_path='yolo26n.pt',           # Private detection model
            pose_model_path='yolo26n-pose.pt', # Private pose model (FIXED!)
            camera_id=self.camera_id           # For state isolation
        )
        print(f"[{self.camera_name}] Detector ready.")
        
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
                time.sleep(0.01)
                continue
            
            self.last_processed_timestamp = timestamp

            # Process Frame
            try:
                processed_frame, status, is_saved = self.detector.process_frame(
                    raw_frame, 
                    frame_count, 
                    skip_frames=3,
                    save_screenshots=True,
                    conf_threshold=self.conf_threshold,
                    camera_name=self.camera_name
                )
                
                self.latest_processed_frame = processed_frame
                self.status = status
                self.last_update_time = time.time()
                
            except Exception as e:
                print(f"[{self.camera_name}] Error in processing: {e}")
                import traceback
                traceback.print_exc()
            
            frame_count += 1
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.01)

        print(f"[{self.camera_name}] Processing thread stopped.")
        self.reader.stop()

    def stop(self):
        self.running = False
        self.join(timeout=2.0)

    def get_frame(self):
        return self.latest_processed_frame

    def get_status(self):
        # If data is stale (> 3 seconds), consider it disconnected
        if time.time() - self.last_update_time > 3.0:
            return "disconnected"
        return self.status


class CameraManager:
    def __init__(self, config_file="cameras.json"):
        """
        FIXED: No shared models strategy.
        Each camera gets private model instances for complete isolation.
        """
        self.config_file = config_file
        self.cameras = {}  # id -> CameraThread
        
        print("=" * 60)
        print("Initializing Camera Manager")
        print("Strategy: Private models per camera (thread-safe)")
        print("=" * 60)
        
        self.load_config_and_start()

    def load_config_and_start(self):
        if not os.path.exists(self.config_file):
            # Default config: Webcam
            default_config = [
                {"id": 0, "name": "Webcam Main", "source": 0}
            ]
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        with open(self.config_file, 'r') as f:
            configs = json.load(f)
            
        # Start threads for each config
        for conf in configs:
            self.add_camera_thread(conf)

    def add_camera_thread(self, config):
        """Add a camera thread with private models."""
        # Convert source to int if it's a digit (for webcam index)
        source = config['source']
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        config['source'] = source

        cam_id = config['id']
        if cam_id in self.cameras:
            print(f"Camera {cam_id} already running.")
            return

        print(f"\n[Camera {cam_id}] Starting camera: {config['name']}")

        # FIXED: No shared models passed - thread creates its own
        thread = CameraThread(config)
        thread.start()
        self.cameras[cam_id] = thread

        print(f"[Camera {cam_id}] Started successfully.\n")

    def add_camera(self, name, source):
        """Add a new camera at runtime."""
        # Generate new ID
        existing_ids = [c.camera_id for c in self.cameras.values()]
        new_id = max(existing_ids) + 1 if existing_ids else 0
        
        new_config = {"id": new_id, "name": name, "source": source}
        
        # Update JSON
        self.save_config_append(new_config)
        
        # Start Thread
        self.add_camera_thread(new_config)

    def remove_camera(self, cam_id):
        """Remove a camera at runtime."""
        if cam_id in self.cameras:
            print(f"Removing camera {cam_id}...")
            self.cameras[cam_id].stop()
            del self.cameras[cam_id]
            
            # Update JSON
            self.save_config_remove(cam_id)
            print(f"Camera {cam_id} removed.")

    def save_config_append(self, new_config):
        """Append new camera config to JSON."""
        try:
            with open(self.config_file, 'r') as f:
                configs = json.load(f)
            configs.append(new_config)
            with open(self.config_file, 'w') as f:
                json.dump(configs, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def save_config_remove(self, cam_id):
        """Remove camera config from JSON."""
        try:
            with open(self.config_file, 'r') as f:
                configs = json.load(f)
            configs = [c for c in configs if c['id'] != cam_id]
            with open(self.config_file, 'w') as f:
                json.dump(configs, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_active_cameras(self):
        """Get dictionary of active camera threads."""
        return self.cameras

    def update_global_conf(self, conf):
        """Update confidence threshold for all cameras."""
        for cam in self.cameras.values():
            cam.conf_threshold = conf
