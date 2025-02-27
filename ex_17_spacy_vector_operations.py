# python -m spacy download de_core_news_sm es gibt auch sm, md, lg (small, medium, large)
import de_core_news_lg
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine

nlp = de_core_news_lg.load()


print("-----Finde die Stellung von Wörtern im Satz:")
doc = nlp("Dies ist ein Satz.")
print([(w.text, w.pos_) for w in doc])


print("-----Finde die Ähnlicheiten von 2 Wörtern:")
token1 = "Lehrer"
token2 = "Pädagoge"

word1 = nlp(token1)
word2 = nlp(token2)


print(word1.has_vector)
print(word1.vector_norm)

print("shape", word1.vector.shape)

# Embeddings als Vektoren ausgeben
print("Embedding von '%s': %s"  % (token1, word1.vector[:10]))  # Zeigt die ersten 10 Werte
print("Embedding von '%s': %s"  % (token2, word2.vector[:10]))

# Ähnlichkeit berechnen (Kosinus-Ähnlichkeit)
similarity = word1.similarity(word2)
print(f"Ähnlichkeit zwischen '%s'und '%s': %.4f"  % (token1, token2, similarity))





# Liste mit Wortpaaren für die Geschlechterdifferenz
gendered_pairs = [
    ("Arzt", "Ärztin"), ("Lehrer", "Lehrerin"), ("Mitarbeiter", "Mitarbeiterin"),
    ("Professor", "Professorin"), ("Schüler", "Schülerin"), ("Kollege", "Kollegin"),
    ("Kunde", "Kundin"), ("Autor", "Autorin"), ("Direktor", "Direktorin"),
    ("Bürger", "Bürgerin"), ("Freund", "Freundin"), ("Chef", "Chefin"),
    ("Gast", "Gästin"), ("Sänger", "Sängerin"), ("Trainer", "Trainerin"),
    ("Dichter", "Dichterin"), ("Arbeiter", "Arbeiterin"), ("Ingenieur", "Ingenieurin"),
    ("Verkäufer", "Verkäuferin"), ("Forscher", "Forscherin"), ("Gärtner", "Gärtnerin"),
    ("Maler", "Malerin"), ("Richter", "Richterin"), ("Redakteur", "Redakteurin"),
    ("Wissenschaftler", "Wissenschaftlerin"), ("Pilot", "Pilotin"), ("Polizist", "Polizistin"),
    ("Psychologe", "Psychologin"), ("Regisseur", "Regisseurin"), ("Sekretär", "Sekretärin"),
    ("Techniker", "Technikerin"), ("Wirt", "Wirtin"), ("Zahnarzt", "Zahnärztin"),
    ("Genosse", "Genossin"), ("Student", "Studentin"), ("Doktor", "Doktorin"),
    ("Therapeut", "Therapeutin"), ("Philosoph", "Philosophin"), ("Bäcker", "Bäckerin"),
]

# Berechne den Geschlechtsvektor als Durchschnitt aller Differenzen
gender_vectors = [nlp(female).vector - nlp(male).vector for male, female in gendered_pairs]
gender_vector = np.mean(gender_vectors, axis=0)  # Mittelwert des Geschlechtsvektors


seek_word ="Verkäuferin"
seek = nlp(seek_word).vector

# Neutralisierung: Ziehe den Geschlechtsvektor von "Arzt" ab
neutral_vector = seek - gender_vector

# Funktion zur Suche des nächstgelegenen Wortes
def closest_word(vector):
    closest_token = min(
        nlp.vocab,
        key=lambda w: cosine(w.vector, vector) if w.has_vector else float('inf')
    )
    return closest_token.text


def closest_words(vector):
    closest_token = min(
        nlp.vocab,
        key=lambda w: cosine(w.vector, vector) if w.has_vector else float('inf')
    )
    return closest_token.text

# Suche das beste geschlechtsneutrale Wort
neutral_word = closest_word(neutral_vector)

print("Geschlechtsneutrale Alternative zu" , seek_word, ":", neutral_word)