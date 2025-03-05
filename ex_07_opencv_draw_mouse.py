import cv2
import numpy as np

# Leeres Bild
image = np.ones((500, 500, 3), dtype=np.uint8) * 255

# Maus-Callback-Funktion
def draw(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 10, (0, 0, 255), -1)  # Roter Punkt

# Fenster erstellen und Maus-Callback setzen
cv2.namedWindow('Malen')
cv2.setMouseCallback('Malen', draw)

while True:
    cv2.imshow('Malen', image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC-Taste beendet
        break

cv2.destroyAllWindows()
