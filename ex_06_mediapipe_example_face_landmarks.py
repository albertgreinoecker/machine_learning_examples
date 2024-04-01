import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect the face mesh
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # mp_drawing.draw_landmarks(
                #     image,
                #     face_landmarks,
                #     mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                for idx, landmark in enumerate(face_landmarks.landmark):
                    # Convert normalized position to pixel position
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])

                    # Draw the landmark number
                    cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

                #
                # Example: Print the nose tip and left eye coordinates
                # Nose tip is landmark 4, left eye outer corner is landmark 33 (using the 468 landmark model)
                nose_tip = face_landmarks.landmark[4]
                left_eye_outer = face_landmarks.landmark[33]
                print(f"Nose Tip: x={nose_tip.x:.2f}, y={nose_tip.y:.2f}, z={nose_tip.z:.2f}")
                print(f"Left Eye Outer: x={left_eye_outer.x:.2f}, y={left_eye_outer.y:.2f}, z={left_eye_outer.z:.2f}")

        # Display the resulting frame
        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()
