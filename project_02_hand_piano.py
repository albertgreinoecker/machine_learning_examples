import os
import urllib.request
import cv2
import mediapipe as mp
import pygame
import numpy as np
import time

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
hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5))

pygame.init()
pygame.mixer.init()

SOUNDS = {
    "c2": pygame.mixer.Sound("c2.wav"),
    "h": pygame.mixer.Sound("h.wav"),
    "a": pygame.mixer.Sound("a.wav"),
    "g": pygame.mixer.Sound("g.wav"),
    "fis": pygame.mixer.Sound("fis.wav"),
    "f": pygame.mixer.Sound("f.wav"),
    "e": pygame.mixer.Sound("e.wav"),
    "d": pygame.mixer.Sound("d.wav"),
    "cis": pygame.mixer.Sound("cis.wav"),
    "c": pygame.mixer.Sound("c.wav")
}

NOTE_COLORS = {
    "c2": (255, 0, 0),
    "h": (255, 0, 255),
    "a": (0, 0, 255),
    "g": (0, 255, 255),
    "fis": (0, 255, 0),
    "f": (255, 255, 0),
    "e": (255, 165, 0),
    "d": (128, 0, 128),
    "cis": (255, 20, 147),
    "c": (75, 0, 130)
}

effects = []
active_notes = set()

def spiele_note(note, position, frame_shape):
    if note in SOUNDS:
        SOUNDS[note].play()
        effects.append({
            'position': position,
            'color': NOTE_COLORS[note],
            'start_time': time.time(),
            'duration': 1.0,
            'frame_shape': frame_shape
        })
        active_notes.add(note)

def draw_effects(frame):
    current_time = time.time()
    to_remove = []

    for i, effect in enumerate(effects):
        elapsed = current_time - effect['start_time']
        if elapsed > effect['duration']:
            to_remove.append(i)
            continue

        cx, cy = effect['position']
        color = effect['color']
        radius = int(50 + 50 * elapsed)
        alpha = 1.0 - (elapsed / effect['duration'])

        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), radius, color, -1)

        inner_radius = int(radius * 0.7)
        cv2.circle(overlay, (cx, cy), inner_radius, [min(255, c + 50) for c in color], -1)
        cv2.circle(overlay, (cx, cy), 10, (255, 255, 255), -1)

        cv2.addWeighted(overlay, alpha * 0.5, frame, 1 - alpha * 0.5, 0, frame)

    for i in sorted(to_remove, reverse=True):
        if i < len(effects):
            note_effect = effects.pop(i)
            for note, color in NOTE_COLORS.items():
                if color == note_effect['color']:
                    if note in active_notes:
                        active_notes.remove(note)
                    break

def draw_piano_keys(frame):
    height, width, _ = frame.shape
    key_width = width // 10
    key_height = height // 6

    cv2.rectangle(frame, (0, height - key_height), (width, height), (240, 240, 240), -1)
    notes = ["c", "d", "e", "f", "fis", "cis", "g", "a", "h", "c2"]

    for i, note in enumerate(notes):
        x = i * key_width
        y = height - key_height
        is_active = note in active_notes

        if "is" in note:
            key_color = (40, 40, 40)
            if is_active:
                key_color = (100, 100, 255)
            cv2.rectangle(frame, (x + 5, y), (x + key_width - 5, y + key_height), key_color, -1)
            text_color = (255, 255, 255)
        else:
            key_color = (255, 255, 255)
            if is_active:
                key_color = (100, 255, 100)
            cv2.rectangle(frame, (x, y), (x + key_width, y + key_height), key_color, -1)
            cv2.rectangle(frame, (x, y), (x + key_width, y + key_height), (0, 0, 0), 2)
            text_color = (0, 0, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, note, (x + 10, y + key_height - 20), font, 0.6, text_color, 2)

        finger_info = ""
        if i < 5:
            hand_label = "L"
            if i == 0:
                finger_info = "kl"
            elif i == 1:
                finger_info = "R"
            elif i == 2:
                finger_info = "M"
            elif i == 3:
                finger_info = "Z"
            elif i == 4:
                finger_info = "D"
        else:
            hand_label = "R"
            if i == 5:
                finger_info = "D"
            elif i == 6:
                finger_info = "Z"
            elif i == 7:
                finger_info = "M"
            elif i == 8:
                finger_info = "R"
            elif i == 9:
                finger_info = "kl"

        finger_label = f"{hand_label}-{finger_info}"
        cv2.putText(frame, finger_label, (x + 10, y + 20), font, 0.4, text_color, 1)

last_finger_positions = {}
played_notes = {"Left": set(), "Right": set()}
THRESHOLD = 0.08

cap = cv2.VideoCapture(0)
custom_style = mp_drawing_styles.get_default_hand_landmarks_style()
custom_connections_style = mp_drawing_styles.get_default_hand_connections_style()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    overlay = np.zeros_like(frame)
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (20, 20, 35), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = hands.detect_for_video(mp_frame, int(time.monotonic() * 1000))

    if result.hand_landmarks:
        for hand_idx, (hand_landmarks, handedness) in enumerate(
                zip(result.hand_landmarks, result.handedness)):
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                HAND_CONNECTIONS,
                custom_style,
                custom_connections_style
            )

            hand_points = []
            for lm in hand_landmarks:
                h, w, c = frame.shape
                px, py = int(lm.x * w), int(lm.y * h)
                hand_points.append((px, py))

            if hand_points:
                hull = cv2.convexHull(np.array(hand_points))
                cv2.drawContours(frame, [hull], 0, (0, 255, 255), 2)

            hand_label = handedness[0].category_name
            finger_tips = [4, 8, 12, 16, 20]
            current_finger_positions = {i: (hand_landmarks[i].x, hand_landmarks[i].y) for i in
                                        finger_tips}

            for i, (x, y) in current_finger_positions.items():
                height, width, _ = frame.shape
                cx, cy = int(x * width), int(y * height)

                finger_color = (255, 0, 0) if hand_label == "Left" else (0, 0, 255)
                cv2.circle(frame, (cx, cy), 10, finger_color, -1)

                if i in last_finger_positions:
                    last_x, last_y = last_finger_positions[i]
                    last_cx, last_cy = int(last_x * width), int(last_y * height)
                    cv2.line(frame, (cx, cy), (last_cx, last_cy), (255, 255, 0), 2)

                    if (last_finger_positions[i][1] - y) > THRESHOLD and i not in played_notes[hand_label]:
                        note_to_play = None

                        if hand_label == "Left":
                            if i == 20:
                                note_to_play = "c"
                            elif i == 16:
                                note_to_play = "d"
                            elif i == 12:
                                note_to_play = "e"
                            elif i == 8:
                                note_to_play = "f"
                            elif i == 4:
                                note_to_play = "fis"
                        else:
                            if i == 4:
                                note_to_play = "cis"
                            elif i == 8:
                                note_to_play = "g"
                            elif i == 12:
                                note_to_play = "a"
                            elif i == 16:
                                note_to_play = "h"
                            elif i == 20:
                                note_to_play = "c2"

                        if note_to_play:
                            spiele_note(note_to_play, (cx, cy), frame.shape)
                            played_notes[hand_label].add(i)

                if i in played_notes[hand_label] and (last_finger_positions[i][1] - y) < THRESHOLD:
                    played_notes[hand_label].remove(i)

            last_finger_positions = current_finger_positions

    draw_effects(frame)
    draw_piano_keys(frame)

    height, width, _ = frame.shape
    title = "Hand Piano"
    font = cv2.FONT_HERSHEY_DUPLEX
    text_size = cv2.getTextSize(title, font, 1.5, 2)[0]
    text_x = (width - text_size[0]) // 2

    cv2.putText(frame, title, (text_x + 2, 72), font, 1.5, (0, 0, 0), 4)
    cv2.putText(frame, title, (text_x, 70), font, 1.5, (100, 200, 255), 2)

    cv2.putText(frame, "'q' zum Beenden", (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Handpiano", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()