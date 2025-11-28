from flair.models import TextClassifier
from flair.data import Sentence

# Vortrainiertes Sentiment-Modell laden
classifier = TextClassifier.load("sentiment")

# Text klassifizieren
sentence = Sentence("Ich liebe NLP mit spaCy!")
classifier.predict(sentence)

# Ergebnis anzeigen
print(sentence.labels)
