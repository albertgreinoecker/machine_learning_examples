import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision, BaseOptions

# Create the options that will be used for FaceStylizer
base_options = BaseOptions(model_asset_path='face_stylizer.task')
options = vision.FaceStylizerOptions(base_options=base_options)


def cartoonize_image_stylizer(img):
    # Create the face stylizer
    with vision.FaceStylizer.create_from_options(options) as stylizer:
        return stylizer.stylize(image)


def cartoonize_image(img, ds_factor=4, sketch_mode=False):
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply median blur
    img_gray = cv2.medianBlur(img_gray, 7)

    # Detect edges in the image and threshold it
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

    # 'mask' is the sketch of the image
    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Resize the image to a smaller size for faster computation
    img_small = cv2.resize(img, None, fx=1.0 / ds_factor, fy=1.0 / ds_factor, interpolation=cv2.INTER_AREA)

    # Apply bilateral filter the image multiple times
    for _ in range(9):
        img_small = cv2.bilateralFilter(img_small, 9, 9, 7)

    # Resize back to the original size
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR)

    # Combine the edge mask with the stylized image
    dst = np.zeros(img_gray.shape)
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    return dst


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Apply cartoon effect to the whole frame
        #cartoon_image = cartoonize_image(image)
        cartoon_image = cartoonize_image_stylizer(image)
        # Display the resulting frame
        cv2.imshow('MediaPipe Face Detection with Stylization', cartoon_image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()
