import chromadb
import os

from openai import api_key

from ids import openai_secret

# ChromaDB-In-Memory-Instanz starten
chroma_client = chromadb.Client()

# Neue Vektordatenbank (Collection) erstellen
collection = chroma_client.create_collection(name="my_embeddings")

# Beispiel-Daten einfügen (Wörter mit Embeddings)
collection.add(
    documents=["König regiert das Land", "Königin hat eine Krone", "Der Präsident leitet das Land"],
    ids=["1", "2", "3"]
)

results = collection.query(
    query_texts=["Wer ist das Staatsoberhaupt?"],
    n_results=2
)
print(results)


from chromadb.utils import embedding_functions
get_embedding = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_secret,
            model_name="text-embedding-ada-002"
        )

embeddings = get_embedding(["Heute", "ist", "Sonne"])
print(embeddings)
print(len(embeddings[0]))


# ChromaDB-Client initialisieren
client2 = chromadb.PersistentClient(path="./chroma_db")
collection2 = client2.get_or_create_collection(name="documents2", metadata={"hnsw:space": "cosine"},  # Optional: Ähnlichkeitssuche mit Cosine-Similarity
    embedding_function=None)  # Wir nutzen OpenAI-Embeddings manuell

# Texte mit OpenAI einbetten und in ChromaDB speichern
documents = ["Ich liebe Python!", "Programmieren ist toll", "Wie funktioniert KI?", "Heute ist Sonne", "Python kann alles"]
for i, text in enumerate(documents):
    embedding = get_embedding(text)
    collection2.add(
        embeddings=embedding[0],
        documents=text,
        ids=str(i)
    )

query_embedding = get_embedding("Java ist aber auch super!")
results2 = collection2.query(query_embeddings=query_embedding, n_results=2)

for doc, score in zip(results2["documents"][0], results2["distances"][0]):
    print(f"Dokument: {doc} (Ähnlichkeit: {score})")



from sentence_transformers import SentenceTransformer
import chromadb

# Hugging Face Modell laden (Lokal!)
hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB Client erstellen
client = chromadb.PersistentClient(path="./chroma_db")

# Collection anlegen (Dimension = 384 für MiniLM)
collection = client.get_or_create_collection(name="hf_documents")

# Beispieltexte
documents = ["Ich liebe Python!", "Programmieren ist toll", "Wie funktioniert KI?"]

# Embeddings erstellen und in ChromaDB speichern
for i, text in enumerate(documents):
    embedding = hf_model.encode(text).tolist()  # NumPy -> Python-Liste
    collection.add(
        embeddings=[embedding],
        documents=[text],
        ids=[str(i)]
    )

print("Texte erfolgreich in ChromaDB gespeichert!")


# Abfrage (Query)
query_text = "Ich mag Coding"
query_embedding = hf_model.encode(query_text).tolist()  # Embedding für die Suche

# ChromaDB durchsuchen
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2  # Anzahl der besten Treffer
)

# Ergebnisse ausgeben
print("Ähnlichste Treffer:")
for doc, score in zip(results["documents"][0], results["distances"][0]):
    print(f"Dokument: {doc} (Ähnlichkeit: {score})")


# ############################################
# Einbinden eines ollama-Modells
# ############################################
import chromadb
import subprocess
import json


# Funktion zur Embedding-Erstellung mit Ollama
# Achtung: Ollama muss installiert sein! Der Download dauert sehr lange (4 GB)
# Alternativen:
# mistral	Schnell & effizient
# llama3	Hohe Genauigkeit
# gemma	Optimiert für Suche
# codellama	Für Code-Suche geeignet
def get_embedding(text, model="mistral"):
    command = f'ollama run {model} "Generate an embedding for: {text}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Extrahiere Embedding-Vektor
    embedding = json.loads(result.stdout)["embedding"]
    return embedding


# ChromaDB Client starten
client = chromadb.PersistentClient(path="./chroma_db")

# Collection anlegen
collection = client.get_or_create_collection(name="ollama_documents")

# Beispieltexte
documents = ["Ich liebe Python!", "Programmieren ist toll", "Wie funktioniert KI?"]

# Embeddings mit Ollama erstellen & in ChromaDB speichern
for i, text in enumerate(documents):
    embedding = get_embedding(text)
    collection.add(
        embeddings=[embedding],
        documents=[text],
        ids=[str(i)]
    )

print("Daten erfolgreich in ChromaDB gespeichert!")


# Suchtext
query_text = "Ich mag Coding"
query_embedding = get_embedding(query_text)

# Ähnlichkeitssuche in ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2  # Anzahl der Treffer
)

# Ergebnisse ausgeben
print("Ähnlichste Treffer:")
for doc, score in zip(results["documents"][0], results["distances"][0]):
    print(f"Dokument: {doc} (Ähnlichkeit: {score})")