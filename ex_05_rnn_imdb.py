from deepface.models.demography.Emotion import tf_version
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM, GRU
import os
import  tensorflow as tf
tf.config.optimizer.set_jit(False)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Do not use the GPU

# --- 1. IMDB-Daten laden ---
# Wir nehmen die 10.000 häufigsten Wörter
max_features = 10000
maxlen = 200           # Wir betrachten nur die ersten 200 Wörter pro Review

print("Lade Daten ...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# --- Wortindex laden ---
word_index = imdb.get_word_index()

X_train[0]

# Das Mapping invertieren: index → wort
reverse_word_index = {value + 3: key for key, value in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"   # unbekanntes Wort
reverse_word_index[3] = "<UNUSED>"

# Beispiel: ersten Review wieder in Text zurückwandeln
decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in X_train[0]])
print("Beispiel-Review (entschlüsselt):", decoded_review)

#0 negativ, 1 positiv
print(y_train[0])

print(len(X_train), "Trainingsbeispiele")
print(len(X_test), "Testbeispiele")

# --- 2. Sequenzen auf gleiche Länge bringen ---
# Kürzere Sequenzen werden mit 0 (dem <PAD>-Token) am Anfang aufgefüllt (Standard).
# Längere Sequenzen werden am Anfang abgeschnitten, sodass nur die letzten maxlen Wörter bleiben.
# Möchte man lieber am Ende abschneiden, dann  padding='post', truncating='post' setzen
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# --- 3. Modell erstellen ---

#Möchte man mehr kombinieren, dann muss man return_sequences=True setzen, z.B.:
#SimpleRNN(32, return_sequences=True),  # gibt Sequenz aus
# SimpleRNN(32),                         # verarbeitet Sequenz weiter

model = Sequential([
    Embedding(max_features, 32, input_length=maxlen),  # Wandelt Wortindizes in Vektoren
    #SimpleRNN(32),                                     # Einfaches RNN mit 32 Einheiten (Neuronen). Jede Zelle hat ihr eigenes Gedächtnis.
    LSTM(64),
    # GRU(64),                                       # GRU-Schicht mit 64 Einheiten
    Dense(1, activation='sigmoid')                     # Binäre Sentiment-Ausgabe (positiv/negativ)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 4. Modell trainieren ---
print("Trainiere Modell ...")
history = model.fit(X_train, y_train,
                    epochs=3,
                    batch_size=64,
                    validation_split=0.2)


#Möchte man sich die Embeddings anschauen:
embedding_layer = model.layers[0]


# --- 5. Modell evaluieren ---
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTestgenauigkeit: {acc:.2f}")


# --- 6. Vorhersagen machen ---

def encode_review(text):
    # Kleinbuchstaben + Wörter trennen
    words = text.lower().split()
    encoded = [1]  # <START> Symbol

    for w in words:
        if w in word_index:
            encoded.append(word_index[w])
        else:
            encoded.append(2)  # <UNK> für unbekanntes Wort
    return encoded



model.save("models/ex_5_imdb.model.h5")  # oder "sentiment_model.keras"

######################################################
from tensorflow.keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Do not use the GPU

model = load_model("models/ex_5_imdb.model.h5")

my_review = (
    "I really loved this movie. The story was engaging and the characters felt real. "
    "The performances were top-notch, especially the lead actor. "
    "The pacing kept me interested from start to finish and the ending was beautiful. "
    "Definitely one of the best films I’ve seen this year."
)
encoded_review = encode_review(my_review[:maxlen])
print(encoded_review)

pred = model.predict(sequence.pad_sequences([encoded_review], maxlen=maxlen))
print(pred)


my_review = (
    "Honestly, this was one of the worst movies I have ever seen. "
    "The plot made no sense, the characters were flat, and the dialogue was painful to listen to. "
    "It felt like the writers had no idea what they wanted to say. "
    "Even the special effects looked cheap and outdated. "
    "I tried to stay positive and give it a chance, but after an hour I was just frustrated. "
    "There wasn’t a single moment that felt genuine or engaging. "
    "Avoid this movie if you value your time."
)
encoded_review = encode_review(my_review[:maxlen])
padded = sequence.pad_sequences([encoded_review], maxlen=maxlen)
print(encoded_review[:maxlen])

pred = model.predict(padded)
print(pred)

