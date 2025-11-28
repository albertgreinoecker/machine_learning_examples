import gensim.downloader as api
from sklearn.decomposition import PCA

# Vortrainiertes Word2Vec-Modell laden (Google News, 1.5GB groß)
#model = api.load("word2vec-google-news-300")
model = api.load("glove-wiki-gigaword-100") #128 MB

words = list(model.key_to_index.keys())
words[:200]

model.get_vector("computer")

# Ähnlichste Wörter zu "king"
print(model.most_similar("king"))
print(model.most_similar("doctor"))
print(model.most_similar("university"))

# Vektoroperation: king - man + woman = queen
result = model.most_similar(positive=['king', 'woman'], negative=['man'])
print(result)

# Wortähnlichketien berechnen
print(model.similarity("king", "queen"))
print(model.similarity("car", "bus"))
print(model.similarity("dog", "cat"))
print(model.similarity("car", "saxophone"))
print(model.similarity("doctor", "nurse"))
print(model.similarity("doctor", "anything"))


#finde Zusammenhänge zwischen Konzepten, z. B.:
print(model.most_similar(positive=["Einstein", "painter"], negative=["scientist"]))
print(model.most_similar(positive=["Paris", "Italy"], negative=["France"]))



# Welche Wörter passen nicht in die Reihe?
words = ["apple", "banana", "grape", "carrot"]
outlier = model.doesnt_match(words)
print(outlier)


# Visualisierung der Word-Embeddings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Liste von Wörtern
words = ["king", "queen", "man", "woman", "dog", "cat", "apple", "banana"]
vectors = np.array([model[word] for word in words])

# PCA-Reduktion auf 2D
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# Visualisierung
plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
    plt.text(reduced_vectors[i, 0]+0.02, reduced_vectors[i, 1]+0.02, word)

plt.show()


# Herauslesen der Embeddings für ein Wort
print(model["king"])
print(len(model["king"]))



# Ähnlichkeiten zwischen Sätze berechnen mittels mean pooling
def sentence_vector(sentence, model):
    words = sentence.lower().split()
    word_vectors = [model[word] for word in words if word in model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

sentence1 = "The cat is sitting on the mat"
#sentence2 = "A dog is lying on the rug"
sentence2 = "Roses are red, violets are blue"

vec1 = sentence_vector(sentence1, model)
vec2 = sentence_vector(sentence2, model)

similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(f"Similarity between sentences: {similarity:.2f}")

# Ähnliche Texte finden

def find_most_similar(query, documents, model):
    query_vec = sentence_vector(query, model)
    doc_vecs = [sentence_vector(doc, model) for doc in documents]

    similarities = [np.dot(query_vec, doc) / (np.linalg.norm(query_vec) * np.linalg.norm(doc)) for doc in doc_vecs]
    return sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)

documents = ["The stock market is rising", "New AI breakthrough in robotics", "Top 10 movies of the year"]
query = "Latest AI trends"

print(find_most_similar(query, documents, model))

