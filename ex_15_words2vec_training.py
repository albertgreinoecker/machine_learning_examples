from gensim.models import Word2Vec
from gensim.test.utils import common_texts  # Beispiel-Datensatz (Liste von Token-Listen)

# 1. Trainingsdaten vorbereiten
#    common_texts ist eine Liste von Listen, z. B.:
#    [
#        ['human', 'interface', 'computer'],
#        ['survey', 'user', 'computer', 'system', ...],
#        ...
#    ]
train_sentences = common_texts

# 2. Word2Vec-Modell trainieren
#    - vector_size: Dimension der erzeugten Word-Embeddings
#    - window: Kontext-Fenstergröße
#    - min_count: minimale Häufigkeit eines Wortes, damit es berücksichtigt wird
#    - workers: Anzahl der parallelen Threads (falls unterstützt)
model = Word2Vec(
    sentences=train_sentences,
    vector_size=50,
    window=5,
    min_count=1,
    workers=4
)

# 3. Zugriff auf die gelernten Word-Embeddings
word_vector_computer = model.wv['computer']
print("Vektor für das Wort 'computer':\n", word_vector_computer)

# 4. Nächste ähnliche Wörter ermitteln
similar_words = model.wv.most_similar('computer', topn=3)
print("\nÄhnlichste Wörter zu 'computer':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")

# 5. Beispiel: Wie ähnlich sind zwei Wörter?
similarity = model.wv.similarity('system', 'user')
print(f"\nÄhnlichkeit zwischen 'system' und 'user': {similarity:.4f}")

