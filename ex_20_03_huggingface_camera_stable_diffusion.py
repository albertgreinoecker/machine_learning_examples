import cv2
import torch
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# Modell für Bild-zu-Bild-Generierung (Stable Diffusion)
MODEL_NAME = "runwayml/stable-diffusion-v1-5"

# Stable Diffusion Pipeline laden
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)

# Kamera-Feed starten
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV Bild in PIL-Format konvertieren
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Prompt für die Cartoon-Umwandlung
    prompt = "A cartoon-style face, Pixar-like, highly detailed, vibrant colors"

    # Verarbeitung mit Stable Diffusion img2img (niedrige Stärken verhindern starke Verzerrung)
    cartoonized_image = pipe(prompt=prompt, image=image_pil, strength=0.5, guidance_scale=7.5).images[0]

    # PIL in OpenCV-Format konvertieren
    cartoonized_image = cv2.cvtColor(np.array(cartoonized_image), cv2.COLOR_RGB2BGR)

    # Cartoon-Bild anzeigen
    cv2.imshow("Cartoon Face", cartoonized_image)

    # Beenden mit 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
