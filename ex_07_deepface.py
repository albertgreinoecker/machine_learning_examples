import cv2
from deepface import DeepFace

# Webcam starten
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Gesichtsanalyse in Echtzeit (nur einmal pro Frame)
    try:
        analysis = DeepFace.analyze(img_path=frame, actions=["emotion", "age", "gender", "race"],
                                    enforce_detection=False)
        emotion = analysis[0]["dominant_emotion"]
        cv2.putText(frame, f"Emotion: {emotion}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        age = analysis[0]["age"]
        cv2.putText(frame, f"Age: {age}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        race = analysis[0]["gender"]
        cv2.putText(frame, f"gender: {race}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        race = analysis[0]["race"]
        idx = 0
        for key, value in race.items():
            cv2.putText(frame, f"{key}: {value}", (30, 150+(idx*30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            idx += 1

    except Exception as e:
        pass  # Falls kein Gesicht erkannt wird, einfach weitermachen
        print(e)

    # Zeige das Bild mit erkannten Emotionen
    cv2.imshow("DeepFace Emotion Detection", frame)

    # Beenden mit 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
