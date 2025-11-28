import nltk
from nltk.util import ngrams
from nltk.corpus import gutenberg

# Lade den Text8-Korpus (nur ein Satz aus Platzgründen)
#nltk.download('text8')
words = gutenberg.words()[:1000]  # Ein Ausschnitt für die Demonstration

# Erzeuge Trigramme
trigrams = list(ngrams(words, 3))

# Zeige einige Beispiele
for gram in trigrams[:10]:
    print(gram)


words = gutenberg.words('melville-moby_dick.txt')
print(len(words))
print(words[:20])