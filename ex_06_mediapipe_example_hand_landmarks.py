import os
import time
import urllib.request

import cv2
import mediapipe as mp

# Download the hand landmark model once (if not present yet)
MODEL_PATH = "models/hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initialize MediaPipe Hand Landmarker
vision = mp.tasks.vision
mp_drawing = vision.drawing_utils
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
options = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Initialize webcam
cap = cv2.VideoCapture(0)

with vision.HandLandmarker.create_from_options(options) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB and wrap it as MediaPipe image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detect hands and hand landmarks (video mode needs a steadily increasing timestamp in ms)
        results = hands.detect_for_video(mp_image, int(time.monotonic() * 1000))

        # Draw the hand annotations on the (BGR) image
        for hand_landmarks in results.hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )

            # Print landmarks for each hand detected
            for id, landmark in enumerate(hand_landmarks):
                print(f"Landmark {id}: x={landmark.x:.2f}, y={landmark.y:.2f}, z={landmark.z:.2f}")

        # Display the resulting frame
        cv2.imshow("MediaPipe Hands", image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()
