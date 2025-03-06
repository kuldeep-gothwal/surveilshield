import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

class TheftDetector:
    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        
        # Initialize YOLO model
        self.yolo_model = YOLO(settings.get('model_path', 'yolov8n.pt'))
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=settings.get('pose_detection_confidence', 0.5),
            min_tracking_confidence=settings.get('pose_tracking_confidence', 0.5)
        )
        
        # Detection thresholds
        self.confidence_threshold = settings.get('confidence_threshold', 0.5)
        self.iou_threshold = settings.get('iou_threshold', 0.5)
        
        # Define target object classes with correct COCO dataset IDs
        # banana=52, cell phone=67, watch=88, pen is custom (you'll need to train for this)
        self.target_object_classes = [52, 67, 88]
        if settings.get('include_pen', True):
            # Assuming you've trained a custom model that includes pen
            self.target_object_classes.append(999)  # custom class ID for pen
            
        # Class name mapping (update with correct class IDs from your model)
        self.class_name_map = {
            0: "person", 
            52: "banana", 
            67: "cell phone", 
            88: "watch",
            999: "pen"  # custom class
        }
        
        # History of detections for temporal smoothing
        self.detection_history = []
        self.history_length = settings.get('history_length', 5)
        
        # Regions of interest definitions (relative to body landmarks)
        self.region_definitions = {
            'upper_chest_pocket': {
                'landmarks': [11, 12],  # shoulders
                'x_offset': 0,
                'y_offset': 0.05,
                'width_factor': 0.15,
                'height_factor': 0.1
            },
            'lower_chest_pocket': {
                'landmarks': [11, 12],  # shoulders
                'x_offset': 0,
                'y_offset': 0.1,
                'width_factor': 0.15,
                'height_factor': 0.1
            },
            'left_waist_pocket': {
                'landmarks': [23, 11],  # left hip, left shoulder
                'x_offset': -0.02,
                'y_offset': 0.1,
                'width_factor': 0.15,
                'height_factor': 0.15
            },
            'right_waist_pocket': {
                'landmarks': [24, 12],  # right hip, right shoulder
                'x_offset': 0.02,
                'y_offset': 0.1,
                'width_factor': 0.15,
                'height_factor': 0.15
            },
            'back_pocket': {
                'landmarks': [23, 24],  # left hip, right hip
                'x_offset': 0,
                'y_offset': 0.1,
                'width_factor': 0.3,
                'height_factor': 0.15
            }
        }

    def detect_objects(self, frame):
        """
        Detect target objects in the frame using YOLO
        """
        results = self.yolo_model(frame)
        detections = []
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                if conf > self.confidence_threshold and class_id in self.target_object_classes:
                    detections.append({
                        'type': self.class_name_map.get(class_id, 'unknown'),
                        'box': box.tolist(),
                        'confidence': float(conf),
                        'class_id': int(class_id),
                        'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                        'area': (box[2] - box[0]) * (box[3] - box[1])
                    })
        
        return detections

    def detect_poses(self, frame):
        """
        Detect body poses in the frame using MediaPipe
        Fixed to handle the missing 'score' field error
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_frame)
        
        if not result.pose_landmarks:
            return []
        
        pose_data = []
        landmarks = result.pose_landmarks.landmark
        
        # Convert landmarks to dictionary format
        landmarks_dict = {i: {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} 
                         for i, lm in enumerate(landmarks)}
        
        # Create pose data without trying to access the score field
        pose_data.append({
            'id': 0,
            'landmarks': landmarks_dict,
            # Don't try to access 'score' field since it doesn't exist
        })
        
        return pose_data

    def calculate_region_of_interest(self, region_def, landmarks, frame_shape):
        """
        Calculate a region of interest based on body landmarks
        """
        frame_w, frame_h = frame_shape[1], frame_shape[0]
        
        # Get reference landmarks
        ref_landmarks = [landmarks.get(lm_id) for lm_id in region_def['landmarks']]
        if any(lm is None for lm in ref_landmarks):
            return None
        
        # Calculate center point
        center_x = sum(lm['x'] for lm in ref_landmarks) / len(ref_landmarks)
        center_y = sum(lm['y'] for lm in ref_landmarks) / len(ref_landmarks)
        
        # Apply offsets
        center_x += region_def['x_offset']
        center_y += region_def['y_offset']
        
        # Convert to pixel coordinates
        center_x = int(center_x * frame_w)
        center_y = int(center_y * frame_h)
        
        # Calculate width and height
        box_width = int(region_def['width_factor'] * frame_w)
        box_height = int(region_def['height_factor'] * frame_h)
        
        # Create bounding box [x1, y1, x2, y2]
        return [
            center_x - box_width // 2, 
            center_y - box_height // 2,
            center_x + box_width // 2, 
            center_y + box_height // 2
        ]

    def calculate_all_regions(self, pose, frame_shape):
        """
        Calculate all defined regions of interest for a pose
        """
        landmarks = pose['landmarks']
        regions = {}
        
        for name, region_def in self.region_definitions.items():
            region_box = self.calculate_region_of_interest(region_def, landmarks, frame_shape)
            if region_box:
                regions[name] = region_box
                
        # Add additional dynamic regions
        
        # Sleeve regions based on elbow and wrist positions
        if all(landmarks.get(lm_id) for lm_id in [13, 15]):  # left elbow, left wrist
            left_elbow = landmarks[13]
            left_wrist = landmarks[15]
            sleeve_x = (left_elbow['x'] + left_wrist['x']) / 2
            sleeve_y = (left_elbow['y'] + left_wrist['y']) / 2
            sleeve_x = int(sleeve_x * frame_shape[1])
            sleeve_y = int(sleeve_y * frame_shape[0])
            sleeve_width = int(0.1 * frame_shape[1])
            sleeve_height = int(0.15 * frame_shape[0])
            regions['left_sleeve'] = [
                sleeve_x - sleeve_width // 2,
                sleeve_y - sleeve_height // 2,
                sleeve_x + sleeve_width // 2,
                sleeve_y + sleeve_height // 2
            ]
            
        if all(landmarks.get(lm_id) for lm_id in [14, 16]):  # right elbow, right wrist
            right_elbow = landmarks[14]
            right_wrist = landmarks[16]
            sleeve_x = (right_elbow['x'] + right_wrist['x']) / 2
            sleeve_y = (right_elbow['y'] + right_wrist['y']) / 2
            sleeve_x = int(sleeve_x * frame_shape[1])
            sleeve_y = int(sleeve_y * frame_shape[0])
            sleeve_width = int(0.1 * frame_shape[1])
            sleeve_height = int(0.15 * frame_shape[0])
            regions['right_sleeve'] = [
                sleeve_x - sleeve_width // 2,
                sleeve_y - sleeve_height // 2,
                sleeve_x + sleeve_width // 2,
                sleeve_y + sleeve_height // 2
            ]
            
        # Upper back region
        if all(landmarks.get(lm_id) for lm_id in [11, 12, 23, 24]):  # shoulders and hips
            back_x = (landmarks[11]['x'] + landmarks[12]['x']) / 2
            back_y = (landmarks[11]['y'] + landmarks[12]['y'] + landmarks[23]['y'] + landmarks[24]['y']) / 4
            back_x = int(back_x * frame_shape[1])
            back_y = int(back_y * frame_shape[0])
            back_width = int(0.2 * frame_shape[1])
            back_height = int(0.25 * frame_shape[0])
            regions['upper_back'] = [
                back_x - back_width // 2,
                back_y - back_height // 2,
                back_x + back_width // 2,
                back_y + back_height // 2
            ]
            
        # Mid back region
        if all(landmarks.get(lm_id) for lm_id in [23, 24]):  # left hip, right hip
            mid_back_x = (landmarks[23]['x'] + landmarks[24]['x']) / 2
            mid_back_y = (landmarks[23]['y'] + landmarks[24]['y']) / 2 - 0.05
            mid_back_x = int(mid_back_x * frame_shape[1])
            mid_back_y = int(mid_back_y * frame_shape[0])
            mid_back_width = int(0.2 * frame_shape[1])
            mid_back_height = int(0.15 * frame_shape[0])
            regions['mid_back'] = [
                mid_back_x - mid_back_width // 2,
                mid_back_y - mid_back_height // 2,
                mid_back_x + mid_back_width // 2,
                mid_back_y + mid_back_height // 2
            ]
        
        return regions

    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union between two bounding boxes
        """
        # Determine the coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        # If there's no intersection, return 0
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        return intersection_area / union_area if union_area > 0 else 0.0

    def is_object_hidden(self, obj_box, regions):
        """
        Determine if an object is hidden in any body region
        """
        for region_name, region_box in regions.items():
            iou = self.calculate_iou(obj_box, region_box)
            if iou > self.iou_threshold:
                return True, region_name
                
        return False, None

    def check_proximity_to_landmarks(self, obj_center, pose, frame_shape):
        """
        Check if an object is in proximity to specific body landmarks
        """
        frame_w, frame_h = frame_shape[1], frame_shape[0]
        landmarks = pose['landmarks']
        obj_x, obj_y = obj_center
        
        # Define landmark IDs and their corresponding body parts
        landmark_regions = {
            0: "nose",
            7: "left ear",
            8: "right ear",
            9: "mouth_left",
            10: "mouth_right",
            15: "left wrist",
            16: "right wrist",
            19: "left hand",
            20: "right hand",
            23: "left hip",
            24: "right hip",
            25: "left knee",
            26: "right knee",
            27: "left ankle",
            28: "right ankle"
        }
        
        # Check proximity to each landmark
        proximity_threshold = 0.08 * max(frame_w, frame_h)  # Dynamic threshold based on frame size
        
        for lm_id, region_name in landmark_regions.items():
            lm = landmarks.get(lm_id)
            if lm:
                lm_x, lm_y = lm['x'] * frame_w, lm['y'] * frame_h
                distance = np.sqrt((lm_x - obj_x)**2 + (lm_y - obj_y)**2)
                if distance < proximity_threshold:
                    return True, region_name
                    
        return False, None

    def detect_theft(self, frame, frame_id):
        """
        Main detection function for theft/hiding behavior
        """
        frame_shape = frame.shape
        detections = self.detect_objects(frame)
        poses = self.detect_poses(frame)
        
        # Initialize results
        results = {
            'frame_id': frame_id,
            'detections': detections,
            'suspicious_actions': [],
            'alert': False
        }
        
        if not poses:
            return results
            
        for pose in poses:
            # Calculate all regions of interest for this pose
            regions = self.calculate_all_regions(pose, frame_shape)
            
            # Check each detected object
            for obj in detections:
                if obj['type'] in ["banana", "cell phone", "watch", "pen"]:
                    # Check if object is in any of the defined regions
                    hidden, location = self.is_object_hidden(obj['box'], regions)
                    
                    if hidden:
                        suspicious_action = {
                            'object': obj['type'],
                            'location': location,
                            'confidence': obj['confidence'],
                            'box': obj['box']
                        }
                        results['suspicious_actions'].append(suspicious_action)
                    else:
                        # Check proximity to landmarks as fallback
                        near, landmark = self.check_proximity_to_landmarks(obj['center'], pose, frame_shape)
                        if near:
                            # Only consider proximity as hiding if near certain body parts
                            if landmark in ["left hip", "right hip", "left hand", "right hand", "left wrist", "right wrist"]:
                                suspicious_action = {
                                    'object': obj['type'],
                                    'location': f"near {landmark}",
                                    'confidence': obj['confidence'],
                                    'box': obj['box']
                                }
                                results['suspicious_actions'].append(suspicious_action)
        
        # Update alert status
        results['alert'] = len(results['suspicious_actions']) > 0
        
        # Add to history for temporal smoothing
        self.detection_history.append(results)
        if len(self.detection_history) > self.history_length:
            self.detection_history.pop(0)
            
        # Apply temporal smoothing (reduce false positives)
        if len(self.detection_history) >= 3:
            # Only alert if we've seen suspicious activity in multiple consecutive frames
            consecutive_alerts = sum(1 for r in self.detection_history[-3:] if r['alert'])
            results['alert'] = consecutive_alerts >= 2
            
        return results

    def annotate_frame(self, frame, results):
        """
        Draw detection results on the frame
        """
        annotated = frame.copy()
        
        # Draw object detections
        for det in results['detections']:
            # Different colors for different object types
            if det['type'] == "person":
                color = (0, 255, 0)  # Green for person
            elif det['type'] == "watch":
                color = (255, 0, 0)  # Blue for watch
            elif det['type'] == "cell phone":
                color = (0, 0, 255)  # Red for phone
            elif det['type'] == "banana":
                color = (0, 255, 255)  # Yellow for banana
            elif det['type'] == "pen":
                color = (255, 0, 255)  # Purple for pen
            else:
                color = (255, 255, 255)  # White for unknown
                
            box = det['box']
            cv2.rectangle(annotated, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         color, 2)
            
            # Draw label
            label = f"{det['type']} {det['confidence']:.2f}"
            cv2.putText(annotated, label,
                       (int(box[0]), int(box[1]) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw suspicious actions
        for i, action in enumerate(results['suspicious_actions']):
            text = f"ALERT: {action['object']} hidden in {action['location']}"
            cv2.putText(annotated, text, 
                       (30, 60 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw box around hidden object with warning color
            if 'box' in action:
                box = action['box']
                # Pulsating effect for alert boxes
                thickness = 3 + (int(time.time() * 5) % 3)
                cv2.rectangle(annotated, 
                             (int(box[0]), int(box[1])), 
                             (int(box[2]), int(box[3])), 
                             (0, 0, 255), thickness)
        
        # Add alert banner if suspicious activity detected
        if results['alert']:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (0, 0, 255), -1)
            cv2.putText(annotated, "SUSPICIOUS ACTIVITY DETECTED", 
                       (annotated.shape[1]//2 - 180, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated

    def process_frame(self, frame, frame_id=0):
        """
        Process a single frame
        """
        results = self.detect_theft(frame, frame_id)
        annotated = self.annotate_frame(frame, results)
        return annotated, results

    def process_video(self, video_path, output_path=None):
        """
        Process a video file
        """
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        
        # Set up video writer if output path is provided
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            annotated, results = self.process_frame(frame, frame_id)
            
            if output_path:
                out.write(annotated)
                
            cv2.imshow('Theft Detection', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_id += 1
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
    def process_test_image(self, image_path):
        """
        Process a single test image
        """
        image = cv2.imread(image_path)
        if image is None:
            return None, None
            
        return self.process_frame(image)

def run_camera_detection():
    """
    Run the theft detector with webcam input
    """
    detector = TheftDetector(settings={
        'confidence_threshold': 0.4,
        'iou_threshold': 0.3,
        'history_length': 5
    })
    
    cap = cv2.VideoCapture(0)
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated, results = detector.process_frame(frame, frame_id)
        
        cv2.imshow('Theft Detection', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_id += 1
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Choose one of the following options:
    
    # Option 1: Run with webcam
    run_camera_detection()
    
    # Option 2: Process a video file
    # detector = TheftDetector()
    # detector.process_video("test_video.mp4", "output_video.mp4")
    
    # Option 3: Process a test image
    # detector = TheftDetector()
    # detector.process_test_image("test_image.jpg")