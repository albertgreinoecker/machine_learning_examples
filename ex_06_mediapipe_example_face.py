import os
import time
import urllib.request

import cv2
import mediapipe as mp

# Download the face detection model once (if not present yet)
MODEL_PATH = "models/blaze_face_short_range.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initialize MediaPipe Face Detection
vision = mp.tasks.vision
mp_drawing = vision.drawing_utils
options = vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    min_detection_confidence=0.5,
)

# Initialize webcam
cap = cv2.VideoCapture(0)

with vision.FaceDetector.create_from_options(options) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB and wrap it as MediaPipe image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detect the faces (video mode needs a steadily increasing timestamp in ms)
        results = face_detection.detect_for_video(mp_image, int(time.monotonic() * 1000))

        # Draw the face detection annotations on the (BGR) image.
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)
            bbox = detection.bounding_box
            print(f"Face Detected with Confidence {detection.categories[0].score:.2f}:")
            print(
                f"  Bounding Box (pixels): x={bbox.origin_x}, y={bbox.origin_y}, width={bbox.width}, height={bbox.height}"
            )
        # Display the resulting frame
        cv2.imshow("MediaPipe Face Detection", image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()
