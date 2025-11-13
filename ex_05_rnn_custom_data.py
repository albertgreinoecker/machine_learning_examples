from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = [
    "Ich trinke gerne Kaffee",
    "Kaffee ist besser als Tee",
    "Ich mag keinen Tee"
]

# Tokenizer erstellen
tokenizer = Tokenizer(num_words=10000, lower=True, oov_token="<UNK>")
tokenizer.fit_on_texts(texts)

# Wortindex ansehen
word_index = tokenizer.word_index
print(word_index)

#Nach der Anwendung können Texte in numerische Sequenzen umwandeln:
sequences = tokenizer.texts_to_sequences(texts)
print(sequences)

# Für die Weiterverarbeitung müssen die Sequenzen oft auf eine einheitliche Länge gebracht werden.
X = pad_sequences(sequences, maxlen=10, padding='post')
print(X)

# So könnte man den Tokenizer speichern (im Bnärformat) und später wieder laden:
import pickle
with open("data/rnn_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)


with open("data/rnn_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)