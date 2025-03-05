import cv2
import numpy as np

# Erstelle ein leeres weißes Bild
#image = np.ones((500, 500, 3), dtype=np.uint8) * 255

#lade ein Bild
image = cv2.imread("data/htl.jpg")
# Neue Größe setzen (Breite, Höhe)
new_size = (500, 500)
image = cv2.resize(image, new_size)


# Zeichne eine Linie (start, end, Farbe, Dicke)
cv2.line(image, (50, 50), (450, 50), (0, 0, 255), 3)  # Rote Linie

# Zeichne ein Rechteck (top-left, bottom-right, Farbe, Dicke)
cv2.rectangle(image, (50, 100), (450, 200), (0, 255, 0), 3)  # Grünes Rechteck

# Zeichne einen Kreis (Mittelpunkt, Radius, Farbe, Dicke)
cv2.circle(image, (250, 350), 50, (255, 0, 0), -1)  # Blauer gefüllter Kreis

# Zeichne einen Text (Text, Position, Font, Skalierung, Farbe, Dicke)
cv2.putText(image, "OpenCV Demo", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Zeichne ein Polygon (Array aus [x,y] Punkten, isClosed, Farbe, Dicke)
pts = np.array([[10, 10], [20, 25], [30, 40], [50, 454]], np.int32)
cv2.polylines(image, [pts], isClosed=True, color=(200, 0, 200), thickness=2)

# Zeichne ein Bild darauf
# Overlay-Bild laden (z. B. PNG mit transparentem Bereich)
overlay = cv2.imread('data/player.png', cv2.IMREAD_UNCHANGED)


####falls ein Alpha-Kanal vorhanden ist, wird dieser gelöscht
# Extrahiere Farbkanäle und Alpha-Kanal
b, g, r, a = cv2.split(overlay)  # a = Alpha-Kanal

# Erstelle das Overlay-Bild ohne Alpha
overlay = cv2.merge((b, g, r))


# Größe anpassen, falls nötig
overlay = cv2.resize(overlay, (50, 50))
# Position des Overlays (oben links)
x, y = 400, 400
h, w, _ = overlay.shape

# Überlagern ohne Transparenz (direkt kopieren)
image[y:y+h, x:x+w] = overlay


# Zeige das Bild
cv2.imshow('Zeichnung', image)
cv2.waitKey(0)
cv2.destroyAllWindows()