import nltk
from nltk import word_tokenize
from nltk.util import bigrams
from collections import defaultdict, Counter
from pathlib import Path
from collections import defaultdict

def train_bigram_model(text):
    """
    Nimmt einen Text (String) und gibt ein Dictionary zurück,
    das Wahrscheinlichkeiten für P(Wort2 | Wort1) speichert.
    """

    # 1. Tokenisierung (Wörter und Satzzeichen extrahieren)
    tokens = word_tokenize(text)

    # 2. Bigrams erzeugen
    bg = list(bigrams(tokens))

    # 3. Häufigkeiten der einzelnen Bigrams zählen
    bigram_counts = Counter(bg)

    # 4. Häufigkeiten der einzelnen (einzelnen) Wörter zählen
    word_counts = Counter(tokens)

    # 5. Dictionary für bedingte Wahrscheinlichkeiten
    #    P(w2 | w1) = bigram_counts[(w1, w2)] / word_counts[w1]
    bigram_probs = {}
    for (w1, w2), count in bigram_counts.items():
        bigram_probs[(w1, w2)] = count / word_counts[w1]

    return bigram_probs



if __name__ == '__main__':

    nltk.download("punkt_tab")

    # Beispiel-Text
    #text = """Die Katze schläft. Die Katze frisst. Die Katze läuft."""
    #text = """Die Katze frisst die Maus. Die Maus frisst den Käse. Die Katze frisst den Käse."""
    text = Path("data/die_verwandlung_de.txt").read_text()
    # Modell trainieren
    bigram_probabilities = train_bigram_model(text)

    # Anzeige der berechneten Wahrscheinlichkeiten
    for pair, prob in bigram_probabilities.items():
        print(f"P({pair[0]} => {pair[1]}) = {prob:.3f}")

    # Umspeichern der Daten so dass die einzelnen Wörter als Schlüssel dienen
    # und die Folgewörter und deren Wahrscheinlichkeiten als Werte
    word_followers = defaultdict(list)
    for pair, prob in bigram_probabilities.items():
        word_followers[pair[0]].append((pair[1], prob))

    print(word_followers)
    print("Followers für 'Gregor':")
    foll = word_followers["Gregor"]
    sorted_foll = sorted(foll, key=lambda x: x[1])
    for f in sorted_foll:
        print(f"  {f[0]}: {f[1]:.3f}")

    print("BETT", word_followers['Bett'])
    print("Musik", word_followers['Musik'])
    print("Tür", word_followers['Tür'])
