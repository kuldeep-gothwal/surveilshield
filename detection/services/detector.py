# detection/services/detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TheftDetector:
    def __init__(self, settings=None):
        self.settings = settings or {}
        
        # Initialize YOLO model for object detection (yolov8n)
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Initialize MediaPipe for pose estimation
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Detection thresholds
        self.confidence_threshold = self.settings.get('confidence_threshold', 0.7)
        self.motion_threshold = self.settings.get('motion_threshold', 500)
        
        # Keep track of person movements (if needed later for advanced logic)
        self.person_tracks = {}
        
        logger.info("Theft detector initialized")
    
    def detect_theft(self, frame, frame_id):
        """Main detection function to analyze a frame for theft behavior."""
        if frame is None:
            return None
        
        # Step 1: Detect people and objects in the frame
        detections = self._detect_objects(frame)
        
        # Step 2: Track or detect human poses
        person_poses = self._detect_poses(frame)
        
        # Step 3: Analyze for suspicious movements
        suspicious_actions = self._detect_suspicious_actions(detections, person_poses)
        
        results = {
            'frame_id': frame_id,
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'suspicious_actions': suspicious_actions,
            'alert': bool(suspicious_actions)  # If there's any suspicious action, raise alert
        }
        
        return results
    
    def _detect_objects(self, frame):
        """Detect people and certain objects using YOLO."""
        # classes=[0, 67] → 0 = person, 67 = cell phone (as an example, can be changed)
        results = self.yolo_model(frame, classes=[0, 67])  
        
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                if confidence > self.confidence_threshold:
                    detections.append({
                        'type': 'person' if class_id == 0 else 'object',
                        'box': box.tolist(),
                        'confidence': float(confidence),
                        'class_id': int(class_id)
                    })
        
        return detections
    
    def _detect_poses(self, frame):
        """Detect human poses using MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_frame)
        
        poses = []
        if pose_results.pose_landmarks:
            # For now, we assume only one person. If multiple people exist, you’d adapt logic.
            for idx, landmarks in enumerate([pose_results.pose_landmarks]):
                pose_data = {
                    'id': idx,
                    'landmarks': {}
                }
                
                if landmarks:
                    for i, landmark in enumerate(landmarks.landmark):
                        pose_data['landmarks'][i] = {
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        }
                
                poses.append(pose_data)
        
        return poses
    
    def _detect_suspicious_actions(self, detections, poses):
        """
        Analyze poses to detect suspicious actions like putting items in pockets.
        This is a placeholder approach.
        """
        suspicious_actions = []
        
        # Focus on hand movements near the hip (pocket area).
        for pose in poses:
            if not pose['landmarks']:
                continue
            
            left_wrist = pose['landmarks'].get(15)
            right_wrist = pose['landmarks'].get(16)
            left_hip = pose['landmarks'].get(23)
            right_hip = pose['landmarks'].get(24)
            
            if (left_wrist and left_hip and 
                self._is_near(left_wrist, left_hip, threshold=0.1)):
                suspicious_actions.append({
                    'type': 'hand_to_pocket',
                    'side': 'left',
                    'confidence': 0.8,
                    'pose_id': pose['id']
                })
            
            if (right_wrist and right_hip and 
                self._is_near(right_wrist, right_hip, threshold=0.1)):
                suspicious_actions.append({
                    'type': 'hand_to_pocket',
                    'side': 'right',
                    'confidence': 0.8,
                    'pose_id': pose['id']
                })
        
        return suspicious_actions
    
    def _is_near(self, landmark1, landmark2, threshold=0.1):
        """Check if two landmarks are close to each other in normalized coordinates."""
        distance = np.sqrt(
            (landmark1['x'] - landmark2['x'])**2 + 
            (landmark1['y'] - landmark2['y'])**2
        )
        return distance < threshold

    def annotate_frame(self, frame, results):
        """Draw bounding boxes and suspicious alerts on the frame for visualization."""
        if not results:
            return frame
        
        annotated = frame.copy()
        
        # Draw bounding boxes
        for detection in results.get('detections', []):
            box = detection['box']
            color = (0, 255, 0) if detection['type'] == 'person' else (255, 0, 0)
            cv2.rectangle(
                annotated, 
                (int(box[0]), int(box[1])), 
                (int(box[2]), int(box[3])), 
                color, 
                2
            )
            
        # Highlight suspicious actions
        if results.get('suspicious_actions'):
            cv2.putText(
                annotated,
                "SUSPICIOUS ACTIVITY DETECTED",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            
        return annotated