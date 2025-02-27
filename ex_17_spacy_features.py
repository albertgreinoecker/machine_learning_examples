import spacy

# Deutsches Modell laden (oder "en_core_web_sm" für Englisch)
nlp = spacy.load("de_core_news_lg") # lg, md, sm

text = """
Angela Merkel war die Bundeskanzlerin von Deutschland. 
Von 2005 bis 2021 war sie im Amt. Davor war Gerhard Schröder Kanzler. 
Unser Direktor ist Herr Helmut Stecher.
"""

doc = nlp(text)
# Erkenne Named Entities
print("----NAMED ENTITIES")
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")


# Wörter vergleichen
word1 = nlp("Hund")
word2 = nlp("Katze")
word3 = nlp("Auto")

print("----WÖRTER VERGLEICHEN")
print(f"Hund vs. Katze: {word1.similarity(word2):.2f}")
print(f"Hund vs. Auto: {word1.similarity(word3):.2f}")

# Satz vergleichen
text1 = nlp("Ich liebe Kaffee.")
text2 = nlp("Ich mag Espresso.")
text3 = nlp("Das Wetter ist schön.")

print("----SATZ VERGLEICHEN")
print(f"Satz 1 vs. Satz 2: {text1.similarity(text2):.2f}")  # Ähnlich
print(f"Satz 1 vs. Satz 3: {text1.similarity(text3):.2f}")  # Nicht ähnlich



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Wörter auswählen
words = ["Hund", "Katze", "Auto", "Fahrrad", "Pferd", "Haus", "Wohnung", "Appartement", "Villa", "Schloss"]
vectors = np.array([nlp(word).vector for word in words])

# PCA für 2D-Reduktion
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# Visualisierung
plt.figure(figsize=(6,4))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))

plt.show()


# Wort Synonyme
nlp = spacy.load("de_core_news_lg")

import spacy
import numpy as np

nlp = spacy.load("de_core_news_md")

def find_synonyms(word, top_n=5):
    token = nlp(word)

    # Sicherstellen, dass das Wort überhaupt einen Vektor hat
    if not token.has_vector:
        print(f"⚠️ Kein Vektor für '{word}' gefunden.")
        return []

    # ALLE Wörter mit Vektoren abrufen (nicht nur das geladene Vocab!)
    words = [nlp.vocab.strings[w] for w in nlp.vocab.vectors]

    # Wortvektoren in ein NumPy-Array umwandeln
    word_vectors = np.array([nlp.vocab[w].vector for w in words])

    # Ähnlichkeiten berechnen (Kosinus-Ähnlichkeit)
    similarities = np.dot(word_vectors, token.vector) / (
        np.linalg.norm(word_vectors, axis=1) * np.linalg.norm(token.vector)
    )

    # Top-N ähnlichste Wörter abrufen (außer dem Wort selbst)
    sorted_indices = np.argsort(similarities)[::-1][1:top_n+1]
    similar_words = [words[i] for i in sorted_indices]

    return similar_words

# 🔥 Test: Synonyme für "glücklich" finden
w = "Tram"
synonyms = find_synonyms(w, top_n=50)
print(f"Synonyme für '{w}': {synonyms}")


