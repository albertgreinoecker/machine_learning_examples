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

vision = mp.tasks.vision
mp_drawing = vision.drawing_utils
mp_drawing_styles = vision.drawing_styles
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
INDEX_FINGER_TIP = 8

# For static images:
IMAGE_FILES = []
image_options = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
)
with vision.HandLandmarker.create_from_options(image_options) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )
        results = hands.detect(mp_image)

        # Print handedness and draw hand landmarks on the image.
        print("Handedness:", results.handedness)
        if not results.hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.hand_landmarks:
            print("hand_landmarks:", hand_landmarks)
            print(
                "Index finger tip coordinates: (",
                f"{hand_landmarks[INDEX_FINGER_TIP].x * image_width}, "
                f"{hand_landmarks[INDEX_FINGER_TIP].y * image_height})",
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        cv2.imwrite("/tmp/annotated_image" + str(idx) + ".png", cv2.flip(annotated_image, 1))
        # Draw hand world landmarks.
        for hand_world_landmarks in results.hand_world_landmarks:
            mp_drawing.plot_landmarks(
                hand_world_landmarks, [(c.start, c.end) for c in HAND_CONNECTIONS], azimuth=5
            )

# For webcam input:
video_options = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
cap = cv2.VideoCapture(0)
with vision.HandLandmarker.create_from_options(video_options) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Convert the BGR image to RGB and wrap it as MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )
        # Video mode needs a steadily increasing timestamp in ms
        results = hands.detect_for_video(mp_image, int(time.monotonic() * 1000))

        # Draw the hand annotations on the (BGR) image.
        for hand_landmarks in results.hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
