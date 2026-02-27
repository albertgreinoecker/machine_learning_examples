import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Lade das Modell (z. B. 'facebook/musicgen-small')
model = MusicGen.get_pretrained("facebook/musicgen-small")

# Eingabetext für die Musikgenerierung
text = ["A relaxing ambient soundscape with birds and water."]

# Generiere die Audiodatei
output = model.generate(text, progress=True)

# Speichere die generierte Musik als WAV-Datei
audio_write("generated_sound", output[0].cpu(), model.sample_rate, format="wav")

print("Sound generiert und gespeichert als 'generated_sound.wav'")
