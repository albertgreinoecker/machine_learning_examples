import os
import time
import urllib.request

import cv2
import mediapipe as mp

# Download the face landmark model once (if not present yet)
MODEL_PATH = "models/face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initialize MediaPipe Face Landmarker (face mesh)
vision = mp.tasks.vision
mp_drawing = vision.drawing_utils
mp_drawing_styles = vision.drawing_styles
options = vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

with vision.FaceLandmarker.create_from_options(options) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB and wrap it as MediaPipe image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detect the face mesh (video mode needs a steadily increasing timestamp in ms)
        results = face_mesh.detect_for_video(mp_image, int(time.monotonic() * 1000))

        # Draw the face mesh annotations on the (BGR) image
        for face_landmarks in results.face_landmarks:
            # mp_drawing.draw_landmarks(
            #     image,
            #     face_landmarks,
            #     vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            for idx, landmark in enumerate(face_landmarks):
                # Convert normalized position to pixel position
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])

                # Draw the landmark number
                cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            #
            # Example: Print the nose tip and left eye coordinates
            # Nose tip is landmark 4, left eye outer corner is landmark 33 (478 landmark model incl. iris)
            #nose_tip = face_landmarks[4]
            #left_eye_outer = face_landmarks[33]
            #print(f"Nose Tip: x={nose_tip.x:.2f}, y={nose_tip.y:.2f}, z={nose_tip.z:.2f}")
            #print(f"Left Eye Outer: x={left_eye_outer.x:.2f}, y={left_eye_outer.y:.2f}, z={left_eye_outer.z:.2f}")

        # Display the resulting frame
        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()
