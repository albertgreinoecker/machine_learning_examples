import base64
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# .env laden
load_dotenv()

# Client initialisieren
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Anfrage an das neue Responses-API
response = client.responses.create(
    model="gpt-4.1-mini",
    input="Gib alle österreichischen Bundesländer als JSON-Liste aus. "
          "Jedes Objekt soll die Felder 'name' und 'capital' enthalten. "
          "Antworte ausschließlich mit gültigem JSON dass man gleich weiterverarbeiten kann, nicht für die Darstellung, also kein Markdown."
)

# Text extrahieren
j_str = response.output_text
print("RAW RESPONSE:")
print(j_str)

# JSON parsen
bundeslaender = json.loads(j_str)

print("\nBundesländer:")
for b in bundeslaender:
    print(f"{b['name']}: {b['capital']}")


########################################
# Bildgenerierung mit DALL·E 3


prompt = "Zeige mir ein Bild von einem Programmierer der zanz im Stil von Don Martin"

result = client.images.generate(
    model="gpt-image-1",
    prompt=prompt,
    size="1024x1024"
)

# Base64 dekodieren
image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Datei speichern
with open("generated_image.png", "wb") as f:
    f.write(image_bytes)

print("Bild gespeichert als generated_image.png")