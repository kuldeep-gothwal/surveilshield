# detection/views.py

import cv2
import time
import os
import logging
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from detection.services.detector import TheftDetector  # Import your detector class

# Configure logging to reduce noise
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Suppress noisy logs from OpenCV
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

def camera_page(request):
    """
    Simple view that renders a page
    with <img src="/camera_feed/"> pointing to the stream.
    """
    return render(request, 'detection/camera_feed.html')


def camera_feed(request):
    """
    Streams annotated frames from your Mac's built-in camera.
    Each frame is analyzed by YOLO + MediaPipe to detect suspicious
    hand-to-pocket movements with reduced logging.
    """
    # Initialize your detection class with improved settings
    theft_detector = TheftDetector(settings={
        'confidence_threshold': 0.5,  # Lower threshold for better detection
        'debug_mode': True,  # Enable debugging information
    })

    def gen_frames():
        # Counter for skipping frames to reduce CPU usage if needed
        frame_counter = 0
        process_every_n_frames = 1  # Process every frame, change to higher number if needed
        
        cap = cv2.VideoCapture(0)  # 0 or 1 depending on your Mac's default camera
        if not cap.isOpened():
            logger.error("Error: Cannot open camera.")
            return

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    logger.warning("Failed to read frame from camera")
                    break

                frame_counter += 1
                
                # Process only every Nth frame to reduce CPU load if needed
                if frame_counter % process_every_n_frames == 0:
                    # Run detection
                    detection_results = theft_detector.detect_theft(frame, frame_counter)

                    # Log only when something significant is detected
                    if detection_results and detection_results.get('suspicious_actions'):
                        logger.warning(">>>>> SUSPICIOUS ACTION DETECTED! <<<<<")
                        
                    # Annotate the frame for visualization
                    annotated_frame = theft_detector.annotate_frame(frame, detection_results)
                else:
                    # Just use the raw frame if we're skipping processing
                    annotated_frame = frame

                # Stream the frame back to the browser
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_data = buffer.tobytes()
                
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n'
                )

        finally:
            cap.release()

    return StreamingHttpResponse(
        gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

# New function for testing with static images (useful for debugging)
def test_image(request):
    """
    Process a test image for debugging purposes.
    Can be accessed via /test_image/ URL.
    """
    test_img_path = request.GET.get('path', 'test_images/test1.jpg')
    
    theft_detector = TheftDetector(settings={
        'confidence_threshold': 0.4,  # Set lower for testing
    })
    
    try:
        # Process the image
        annotated, results = theft_detector.process_test_image(test_img_path)
        
        if annotated is None:
            return JsonResponse({'error': f'Could not process image: {test_img_path}'})
        
        # Save the annotated image
        output_path = f"test_images/result_{os.path.basename(test_img_path)}"
        cv2.imwrite(output_path, annotated)
        
        return JsonResponse({
            'success': True,
            'output_path': output_path,
            'detections': len(results.get('detections', [])),
            'suspicious_actions': results.get('suspicious_actions', [])
        })
        
    except Exception as e:
        logger.exception("Error processing test image")
        return JsonResponse({'error': str(e)})