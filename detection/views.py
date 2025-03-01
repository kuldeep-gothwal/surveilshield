# detection/views.py

import cv2
import time
from django.http import StreamingHttpResponse
from django.shortcuts import render
from detection.services.detector import TheftDetector  # <-- Import your YOLO+MP class

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
    hand-to-pocket movements.
    """
    # Initialize your detection class
    # Optionally pass in custom settings like confidence_threshold, etc.
    theft_detector = TheftDetector(settings={
        'confidence_threshold': 0.7,
        'motion_threshold': 500,
    })

    frame_count = 0

    def gen_frames():
        cap = cv2.VideoCapture(0)  # 0 or 1 depending on your Mac's default camera
        if not cap.isOpened():
            print("Error: Cannot open camera.")
            return

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_count_nonlocal = gen_frames.counter
                gen_frames.counter += 1

                # ------------------------------------------------------------
                # 1) Run YOLO + MediaPipe detection
                # ------------------------------------------------------------
                detection_results = theft_detector.detect_theft(frame, frame_count_nonlocal)

                # detection_results will have something like:
                # {
                #   'frame_id': <num>,
                #   'timestamp': ...,
                #   'detections': [...],
                #   'suspicious_actions': [...],
                #   'alert': True/False
                # }

                # ------------------------------------------------------------
                # 2) If there's suspicious activity, log or handle it
                #    For now, just print to console. In production, you could
                #    create an alert, save to DB, send notification, etc.
                # ------------------------------------------------------------
                if detection_results and detection_results.get('alert'):
                    print(">>>>> SUSPICIOUS ACTION DETECTED! <<<<<")
                    # Example: create alert or call your AlertManager here
                    # alert_manager.create_alert(...)

                # ------------------------------------------------------------
                # 3) Annotate the frame for visualization
                # ------------------------------------------------------------
                annotated_frame = theft_detector.annotate_frame(frame, detection_results)

                # ------------------------------------------------------------
                # 4) Stream the annotated frame to the browser
                # ------------------------------------------------------------
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_data = buffer.tobytes()
                
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n'
                )

        finally:
            cap.release()

    # Attach a counter to the generator function
    gen_frames.counter = 0

    return StreamingHttpResponse(
        gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )