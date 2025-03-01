# detection/services/video_processor.py
import cv2
import os
import time
import threading
import logging
from datetime import datetime
from .detector import TheftDetector
from .alert_manager import AlertManager

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, camera_id, camera_url, settings=None):
        self.camera_id = camera_id
        self.camera_url = camera_url
        self.settings = settings or {}
        
        # Initialize components
        self.detector = TheftDetector(settings)
        self.alert_manager = AlertManager()
        
        # Video processing settings
        self.frame_skip = self.settings.get('frame_skip', 5)  # Process every Nth frame
        self.recording_dir = self.settings.get('recording_dir', 'media/recordings')
        self.alert_cooldown = self.settings.get('alert_cooldown', 30)  # Seconds between alerts
        
        # State variables
        self.is_running = False
        self.last_alert_time = 0
        self.current_frame = None
        self.frame_count = 0
        self.alert_active = False
        self.alert_frames = []
        
        # Ensure directories exist
        os.makedirs(self.recording_dir, exist_ok=True)
        
        logger.info(f"Video processor initialized for camera {camera_id}")
    
    def start(self):
        """Start video processing in a separate thread."""
        if self.is_running:
            return
        self.is_running = True
        
        self.processing_thread = threading.Thread(target=self._process_video)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info(f"Started video processing for camera {self.camera_id}")
        
    def stop(self):
        """Stop video processing."""
        self.is_running = False
        logger.info(f"Stopped video processing for camera {self.camera_id}")
    
    def _process_video(self):
        """Main video processing loop."""
        cap = cv2.VideoCapture(self.camera_url)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video stream for camera {self.camera_id}")
            self.is_running = False
            return
        
        try:
            while self.is_running:
                success, frame = cap.read()
                
                if not success:
                    logger.warning(f"Failed to read frame from camera {self.camera_id}")
                    time.sleep(1)
                    continue
                
                self.current_frame = frame
                self.frame_count += 1
                
                # Process only every Nth frame to reduce CPU usage
                if self.frame_count % self.frame_skip == 0:
                    self._analyze_frame(frame)
                
                # Small delay to prevent 100% CPU usage
                time.sleep(0.01)
                
        finally:
            cap.release()
            logger.info(f"Released video capture for camera {self.camera_id}")
    
    def _analyze_frame(self, frame):
        """Analyze a single frame for theft detection."""
        results = self.detector.detect_theft(frame, self.frame_count)
        
        if results is None:
            return
            
        current_time = time.time()
        if (results.get('alert') and 
            (current_time - self.last_alert_time) > self.alert_cooldown):
            
            # We are in "alert collecting" mode
            if not self.alert_active:
                self.alert_active = True
                self.alert_frames = []
                
            annotated_frame = self.detector.annotate_frame(frame, results)
            self.alert_frames.append(annotated_frame)
            
            # If we've collected enough frames, create an alert
            if len(self.alert_frames) >= 30:  # ~3 seconds at 10 fps
                self._create_alert(results)
                self.alert_active = False
                self.last_alert_time = current_time
        
        elif self.alert_active and not results.get('alert'):
            # We were collecting frames, but no longer detecting suspicious activity
            annotated_frame = self.detector.annotate_frame(frame, results)
            self.alert_frames.append(annotated_frame)
            
            if len(self.alert_frames) >= 50:  # cap at ~5 seconds
                self._create_alert(results)
                self.alert_active = False
    
    def _create_alert(self, detection_results):
        """Create an alert with video evidence."""
        if not self.alert_frames:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(
            self.recording_dir,
            f"alert_{self.camera_id}_{timestamp}.mp4"
        )
        
        self._save_video_clip(video_path)
        
        # Create alert in DB
        alert_data = {
            'camera_id': self.camera_id,
            'timestamp': datetime.now(),
            'detection_results': detection_results,
            'video_evidence': video_path
        }
        
        self.alert_manager.create_alert(alert_data)
        logger.info(f"Created theft alert for camera {self.camera_id}")
    
    def _save_video_clip(self, output_path):
        """Save collected frames as a .mp4 clip."""
        if not self.alert_frames:
            return
            
        height, width = self.alert_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 10, (width, height))
        
        for frame in self.alert_frames:
            writer.write(frame)
            
        writer.release()
        logger.info(f"Saved alert video clip to {output_path}")