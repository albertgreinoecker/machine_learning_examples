import os
import random
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

# Download the hand landmark model once (if not present yet)
MODEL_PATH = "models/hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Setup für Hand-Tracking und Video
vision = mp.tasks.vision
hand_tracking = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
)
video_capture = cv2.VideoCapture(0)

# Variablen
punktestand = 0
spielpunkte = []
letzte_spawn_zeit = time.time()
spawn_intervall = 2.0
spiel_start_zeit = time.time()


def laserschwerter_zeichnen(bild, hand_landmarks, hand_seite):
    if hand_landmarks:
        for hand_landmark, hand_handedness in zip(hand_landmarks, hand_seite):
            # Farbe je nach Handseite
            farbe = (0, 0, 255) if hand_handedness[0].category_name == "Right" else (255, 0, 0)

            # Koordinaten für Handgelenk und Zeigefinger
            handgelenk_x = int(hand_landmark[0].x * bild.shape[1])
            handgelenk_y = int(hand_landmark[0].y * bild.shape[0])
            zeigefinger_x = int(hand_landmark[5].x * bild.shape[1])
            zeigefinger_y = int(hand_landmark[5].y * bild.shape[0])

            # Laserschwert Griff
            cv2.line(
                bild, (handgelenk_x, handgelenk_y), (zeigefinger_x, zeigefinger_y), (0, 0, 0), 10
            )

            # Laserstrahl berechnen und verlängern
            richtung_x = zeigefinger_x - handgelenk_x
            richtung_y = zeigefinger_y - handgelenk_y
            erweiterter_x = zeigefinger_x + 2 * richtung_x
            erweiterter_y = zeigefinger_y + 2 * richtung_y

            # Farbiger Laserstrahl
            cv2.line(
                bild, (zeigefinger_x, zeigefinger_y), (erweiterter_x, erweiterter_y), farbe, 10
            )

            yield (
                handgelenk_x,
                handgelenk_y,
                zeigefinger_x,
                zeigefinger_y,
                erweiterter_x,
                erweiterter_y,
                farbe,
            )


def punkte_zeichnen(bild, spielpunkte, hand_landmarks, hand_seite, geschwindigkeits_multiplikator):
    global punktestand
    hoehe, breite, _ = bild.shape

    zu_entfernende_punkte = []

    for punkt in spielpunkte:
        mittel_x = breite // 2
        mittel_y = hoehe // 2

        # Bewegung berechnen
        dx = (mittel_x - punkt["x"]) * 0.1
        dy = (mittel_y - punkt["y"]) * 0.1

        basis_bewegung = 0.005 * geschwindigkeits_multiplikator
        y_abhaengige_bewegung = (punkt["y"] / hoehe) * 0.02 * geschwindigkeits_multiplikator

        punkt["x"] += dx * basis_bewegung * 3  # Horizontale Bewegung
        punkt["y"] += (
            dy * basis_bewegung + (0.5 * geschwindigkeits_multiplikator) + 0.75
        )  # Vertikale Bewegung

        punkt["size"] += 0.4 * geschwindigkeits_multiplikator  # Ball wird größer

        # Punkt zeichnen mit Glanz-Effekt
        cv2.circle(bild, (int(punkt["x"]), int(punkt["y"])), int(punkt["size"]), (0, 0, 0), -1)
        cv2.circle(
            bild, (int(punkt["x"]), int(punkt["y"])), int(punkt["size"]) - 2, punkt["color"], -1
        )

        # Glanz-Effekt
        glanz_x = int(punkt["x"] - punkt["size"] * 0.3)
        glanz_y = int(punkt["y"] - punkt["size"] * 0.3)
        glanz_groesse = max(2, int(punkt["size"] * 0.4))
        cv2.circle(bild, (glanz_x, glanz_y), glanz_groesse, (255, 255, 255), -1)

        # Punkte entfernen wenn sie aus dem Bild verschwinden
        if punkt["y"] > hoehe or punkt["size"] > 50:
            zu_entfernende_punkte.append(punkt)
            continue

        # Kollisionserkennung
        if punkt["size"] > 10:
            for hand_landmark, hand_handedness in zip(hand_landmarks, hand_seite):
                handgelenk_x = int(hand_landmark[0].x * bild.shape[1])
                handgelenk_y = int(hand_landmark[0].y * bild.shape[0])

                # Kollision mit Handgelenk
                if (handgelenk_x - punkt["x"]) ** 2 + (handgelenk_y - punkt["y"]) ** 2 < punkt[
                    "size"
                ] ** 2:
                    if (
                        punkt["color"] == (0, 0, 255)
                        and hand_handedness[0].category_name == "Right"
                    ) or (
                        punkt["color"] == (255, 0, 0) and hand_handedness[0].category_name == "Left"
                    ):
                        zu_entfernende_punkte.append(punkt)
                        punktestand += 1
                        break

            # Kollision mit Laserschwert
            for schwert in laserschwerter_zeichnen(bild, hand_landmarks, hand_seite):
                _, _, zeigefinger_x, zeigefinger_y, erweiterter_x, erweiterter_y, farbe = schwert
                if punkt["color"] == farbe:
                    linien_laenge = np.sqrt(
                        (erweiterter_x - zeigefinger_x) ** 2 + (erweiterter_y - zeigefinger_y) ** 2
                    )
                    t = (
                        (punkt["x"] - zeigefinger_x) * (erweiterter_x - zeigefinger_x)
                        + (punkt["y"] - zeigefinger_y) * (erweiterter_y - zeigefinger_y)
                    ) / (linien_laenge**2)
                    t = max(0, min(1, t))

                    naechster_x = zeigefinger_x + t * (erweiterter_x - zeigefinger_x)
                    naechster_y = zeigefinger_y + t * (erweiterter_y - zeigefinger_y)

                    abstand = np.sqrt(
                        (punkt["x"] - naechster_x) ** 2 + (punkt["y"] - naechster_y) ** 2
                    )

                    if abstand < punkt["size"]:
                        zu_entfernende_punkte.append(punkt)
                        punktestand += 1
                        cv2.line(
                            bild,
                            (int(punkt["x"] - punkt["size"]), int(punkt["y"] - punkt["size"])),
                            (int(punkt["x"] + punkt["size"]), int(punkt["y"] + punkt["size"])),
                            (255, 255, 255),
                            2,
                        )
                        break

    # Getroffene Punkte entfernen
    for punkt in zu_entfernende_punkte:
        if punkt in spielpunkte:
            spielpunkte.remove(punkt)


# Hauptspielschleife
while video_capture.isOpened():
    erfolg, bild = video_capture.read()
    if not erfolg:
        break

    # Bild spiegeln
    bild = cv2.flip(bild, 1)
    hoehe, breite, _ = bild.shape

    # Geschwindigkeit erhöhen
    vergangene_zeit = time.time() - spiel_start_zeit
    geschwindigkeits_multiplikator = 1.0 + (vergangene_zeit / 60.0)
    aktuelles_spawn_intervall = max(1.0, spawn_intervall / (1 + vergangene_zeit / 120.0))

    # Hand Tracking
    bild_rgb = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)
    mp_bild = mp.Image(image_format=mp.ImageFormat.SRGB, data=bild_rgb)
    ergebnisse = hand_tracking.detect_for_video(mp_bild, int(time.monotonic() * 1000))

    hand_landmarks = ergebnisse.hand_landmarks
    hand_seite = ergebnisse.handedness

    # Laserschwerter zeichnen
    list(laserschwerter_zeichnen(bild, hand_landmarks, hand_seite))

    # Punkte zeichnen
    punkte_zeichnen(bild, spielpunkte, hand_landmarks, hand_seite, geschwindigkeits_multiplikator)

    # Neue Punkte erzeugen
    aktuelle_zeit = time.time()
    if aktuelle_zeit - letzte_spawn_zeit > aktuelles_spawn_intervall:
        spawn_x = random.randint(int(breite * 0.2), int(breite * 0.8))
        spawn_y_position = int(hoehe * 0.6)

        neuer_punkt = {
            "x": spawn_x,
            "y": spawn_y_position,
            "size": 5,
            "color": (0, 0, 255) if random.random() > 0.5 else (255, 0, 0),
        }
        spielpunkte.append(neuer_punkt)
        letzte_spawn_zeit = aktuelle_zeit

    # Infoleiste unten
    infoleiste_hoehe = 50
    cv2.rectangle(bild, (0, hoehe - infoleiste_hoehe), (breite, hoehe), (0, 0, 0), -1)
    cv2.putText(
        bild,
        f"Punkte: {punktestand}",
        (breite - 150, hoehe - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Geschwindigkeit anzeigen
    cv2.putText(
        bild,
        f"Speed: {geschwindigkeits_multiplikator:.1f}x",
        (10, hoehe - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Bild anzeigen
    cv2.imshow("Laser-Schwerter Spiel", bild)

    # Spiel beenden
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
