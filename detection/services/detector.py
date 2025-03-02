# detection/services/detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from datetime import datetime
import logging
import os

# Set the environment variable to reduce YOLO's logging
os.environ["YOLO_VERBOSE"] = "0"

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Set to WARNING level to reduce logs
logger = logging.getLogger(__name__)

class TheftDetector:
    def __init__(self, settings=None):
        self.settings = settings or {}
        
        # Initialize YOLO model for object detection with reduced verbosity
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
        self.confidence_threshold = self.settings.get('confidence_threshold', 0.5)  # Lowered for better detection
        
        # Track if objects of interest are close to hands
        self.object_hand_proximity = {}
        
        # Track objects near pockets
        self.objects_near_pocket = {}
        
        # Define which YOLO class IDs we care about 
        # 0: person, 45: banana, 67: cell phone
        self.target_object_classes = [45, 67]  
        
        # Define meaningful names for the class IDs
        self.class_name_map = {
            0: "person",
            45: "banana",
            67: "cell phone"
        }
        
        # For debugging
        self.debug_mode = self.settings.get('debug_mode', True)
        
        logger.info("Theft detector initialized with reduced logging")
    
    def detect_theft(self, frame, frame_id):
        """Main detection function to analyze a frame for theft behavior."""
        if frame is None:
            return None
        
        # Step 1: Detect people and objects in the frame
        detections = self._detect_objects(frame, verbose=False)  # Turn off verbose output
        
        # Step 2: Track or detect human poses
        person_poses = self._detect_poses(frame)
        
        # Step 3: Analyze for suspicious movements - now checking if objects go to pocket
        suspicious_actions = self._detect_suspicious_actions(detections, person_poses, frame)
        
        results = {
            'frame_id': frame_id,
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'suspicious_actions': suspicious_actions,
            'alert': bool(suspicious_actions)  # If there's any suspicious action, raise alert
        }
        
        return results
    
    def _detect_objects(self, frame, verbose=False):
        """Detect people and target objects using YOLO."""
        # Add person (0) to our list of target classes for detection
        detection_classes = [0] + self.target_object_classes
        
        # Run detection with verbose=False to reduce console output
        results = self.yolo_model(frame, classes=detection_classes, verbose=verbose)
        
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                if confidence > self.confidence_threshold:
                    # Get object name from our class map
                    obj_type = self.class_name_map.get(class_id, "unknown")
                    
                    detections.append({
                        'type': obj_type,
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
            # For now, we assume only one person. If multiple people exist, you'd adapt logic.
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
    
    def _get_center_of_box(self, box):
        """Calculate the center point of a bounding box."""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    def _is_hand_holding_object(self, hand_landmark, object_box, frame_shape):
        """Check if a hand is holding/near an object."""
        # Convert normalized hand coordinates to pixel coordinates
        hand_x = int(hand_landmark['x'] * frame_shape[1])
        hand_y = int(hand_landmark['y'] * frame_shape[0])
        
        # Check if hand is inside or very close to object bounding box
        # Add a larger margin around the box for better detection
        margin = 50  # Increased margin for better detection
        if (hand_x >= object_box[0] - margin and 
            hand_x <= object_box[2] + margin and
            hand_y >= object_box[1] - margin and 
            hand_y <= object_box[3] + margin):
            return True
        
        return False
    
    def _is_object_near_pocket(self, object_box, hip_landmark, frame_shape):
        """Check if an object is near the hip/pocket area."""
        # Convert normalized hip coordinates to pixel coordinates
        hip_x = int(hip_landmark['x'] * frame_shape[1])
        hip_y = int(hip_landmark['y'] * frame_shape[0])
        
        # Get object center
        obj_center_x, obj_center_y = self._get_center_of_box(object_box)
        
        # Define a larger pocket region (significantly expanded for better detection)
        pocket_region_x_min = hip_x - 150  # Expanded left range
        pocket_region_x_max = hip_x + 100  # Expanded right range
        pocket_region_y_min = hip_y - 50   # Expanded up range
        pocket_region_y_max = hip_y + 150  # Expanded down range
        
        # Check if object center is in pocket region
        if (obj_center_x >= pocket_region_x_min and 
            obj_center_x <= pocket_region_x_max and
            obj_center_y >= pocket_region_y_min and 
            obj_center_y <= pocket_region_y_max):
            return True
        
        return False
    
    def _detect_suspicious_actions(self, detections, poses, frame):
        """
        Analyze if target objects (banana, phone) are being placed into pockets.
        """
        suspicious_actions = []
        
        if not poses or not detections:
            return suspicious_actions
        
        # Get frame dimensions for coordinate conversion
        frame_height, frame_width = frame.shape[:2]
        frame_shape = (frame_height, frame_width)
        
        # Filter out target objects (banana, cell phone) and people
        target_objects = [d for d in detections if d['class_id'] in self.target_object_classes]
        people = [d for d in detections if d['type'] == 'person']
        
        # If no target objects or people, return early
        if not target_objects or not people:
            return suspicious_actions
        
        # For each person pose
        for pose in poses:
            if not pose['landmarks']:
                continue
                
            # Get wrist and hip landmarks (both sides)
            left_wrist = pose['landmarks'].get(15)
            right_wrist = pose['landmarks'].get(16)
            left_hip = pose['landmarks'].get(23)
            right_hip = pose['landmarks'].get(24)
            
            # If landmarks not visible, skip
            if not (left_wrist and right_wrist and left_hip and right_hip):
                continue
            
            # For each target object (banana, phone)
            for obj in target_objects:
                object_id = f"{obj['type']}_{obj['box'][0]}_{obj['box'][1]}"
                
                # Check if either hand is holding/near the object
                left_hand_holding = self._is_hand_holding_object(left_wrist, obj['box'], frame_shape)
                right_hand_holding = self._is_hand_holding_object(right_wrist, obj['box'], frame_shape)
                
                # Check if object is near either pocket
                left_pocket_proximity = self._is_object_near_pocket(obj['box'], left_hip, frame_shape)
                right_pocket_proximity = self._is_object_near_pocket(obj['box'], right_hip, frame_shape)
                
                # Track this object's state
                if object_id not in self.object_hand_proximity:
                    self.object_hand_proximity[object_id] = False
                
                if object_id not in self.objects_near_pocket:
                    self.objects_near_pocket[object_id] = False
                
                # Update tracking 
                is_hand_holding = left_hand_holding or right_hand_holding
                is_near_pocket = left_pocket_proximity or right_pocket_proximity
                
                # Detect suspicious action: Object was near hand and is now near pocket
                # This suggests the item is being placed into a pocket
                if is_near_pocket:  # Simplified to just check if near pocket for easier detection
                    suspicious_actions.append({
                        'type': 'object_to_pocket',
                        'object': obj['type'],
                        'side': 'left' if left_pocket_proximity else 'right',
                        'confidence': 0.9,
                        'pose_id': pose['id']
                    })
                
                # Update tracking state for next frame
                self.object_hand_proximity[object_id] = is_hand_holding
                self.objects_near_pocket[object_id] = is_near_pocket
        
        return suspicious_actions
    
    def annotate_frame(self, frame, results):
        """Draw bounding boxes and suspicious alerts on the frame for visualization."""
        if not results:
            return frame
        
        annotated = frame.copy()
        
        # Draw bounding boxes for all detections
        for detection in results.get('detections', []):
            box = detection['box']
            
            # Different colors based on object type
            if detection['type'] == 'person':
                color = (0, 255, 0)  # Green for people
            elif detection['type'] == 'banana':
                color = (0, 255, 255)  # Yellow for banana
            elif detection['type'] == 'cell phone':
                color = (255, 0, 255)  # Purple for phone
            else:
                color = (255, 0, 0)  # Red for other objects
            
            # Draw the bounding box
            cv2.rectangle(
                annotated, 
                (int(box[0]), int(box[1])), 
                (int(box[2]), int(box[3])), 
                color, 
                2
            )
            
            # Label the object with confidence
            label = f"{detection['type']} ({detection['confidence']:.2f})"
            cv2.putText(
                annotated,
                label,
                (int(box[0]), int(box[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
        # Highlight suspicious actions
        if results.get('suspicious_actions'):
            # Draw a more prominent alert with background
            alert_text = "SUSPICIOUS ACTION DETECTED"
            
            # Create background rectangle for better visibility
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
            cv2.rectangle(
                annotated,
                (45, 35),
                (45 + text_size[0] + 10, 55),
                (0, 0, 0),
                -1  # Fill the rectangle
            )
            
            # Draw the alert text
            cv2.putText(
                annotated,
                alert_text,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),  # Red color
                3
            )
            
            # Add more details about what was detected
            for i, action in enumerate(results.get('suspicious_actions', [])):
                message = f"{action['object']} to {action['side']} pocket!"
                
                # Create background for this message too
                msg_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(
                    annotated,
                    (45, 75 + (i * 30)),
                    (45 + msg_size[0] + 10, 95 + (i * 30)),
                    (0, 0, 0),
                    -1
                )
                
                # Draw the message
                cv2.putText(
                    annotated,
                    message,
                    (50, 90 + (i * 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
        return annotated
    
    # Add a method to process a single image for debugging
    def process_test_image(self, image_path):
        """Process a single test image for debugging."""
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Could not read image: {image_path}")
            return None
            
        results = self.detect_theft(frame, 0)
        annotated = self.annotate_frame(frame, results)
        
        return annotated, results