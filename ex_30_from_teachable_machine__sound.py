import numpy as np
import sounddevice as sd
from tensorflow.lite.python import interpreter as tflite

# === CONFIG ===
SAMPLE_RATE = 44100
DURATION = 1  # Sekunden
MODEL_PATH = "models/soundclassifier_with_metadata.tflite"
LABELS_PATH = "models/labels_soundclassifier.txt"
def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    print("Aufnahme läuft...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Aufnahme abgeschlossen.")
    return np.squeeze(audio)

def preprocess_raw_audio(audio, input_shape):
    # ggf. trimmen oder auffüllen
    target_length = input_shape[1]
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    return np.expand_dims(audio, axis=0).astype(np.float32)  # shape: (1, target_length)

def load_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def predict(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

if __name__ == "__main__":
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']  # z. B. (1, 44100)

    labels = load_labels(LABELS_PATH)

    audio = record_audio()
    audio_input = preprocess_raw_audio(audio, input_shape)
    prediction = predict(interpreter, audio_input)

    for i, prob in enumerate(prediction):
        print(f"{labels[i]}: {prob:.2%}")