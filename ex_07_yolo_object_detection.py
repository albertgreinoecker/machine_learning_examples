import cv2
from ultralytics import YOLO

# YOLOv8 Modell laden (vortrainiertes Modell yolov8n.pt nutzen)
model = YOLO("yolov8n.pt")

# Webcam starten
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Objekterkennung mit YOLO
    results = model(frame)

    # Ergebnisse auf das Frame zeichnen
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding Box Koordinaten
            conf = float(box.conf[0])  # Konfidenzwert
            cls = int(box.cls[0])  # Klassenindex
            label = model.names[cls]  # Klassenname

            # Bounding Box zeichnen
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Zeige das Bild mit den erkannten Objekten
    cv2.imshow("YOLOv8 Live Detection", frame)

    # Beenden mit 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
