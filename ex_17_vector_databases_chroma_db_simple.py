import chromadb
from chromadb.utils import embedding_functions

'''
Simple example of using ChromaDB vector database to store documents with metadata.
'''

documents = [
    "Normalisierung reduziert Redundanz in relationalen Datenbanken.",
    "Ein Index beschleunigt Suchabfragen in Datenbanksystemen.",
    "Python ist eine interpretierte Programmiersprache.",
    "SQL wird für relationale Datenbanken verwendet.",
    "Machine Learning nutzt Daten zur Modellanpassung."
]

metadatas = [
    {"topic": "datenbanken", "type": "definition"},
    {"topic": "datenbanken", "type": "erklaerung"},
    {"topic": "programmiersprachen", "type": "definition"},
    {"topic": "datenbanken", "type": "definition"},
    {"topic": "ki", "type": "definition"}
]

ids = [f"doc_{i}" for i in range(len(documents))]


'''
Initialize ChromaDB client and create a collection with SentenceTransformer embeddings.
'''



client = chromadb.Client()

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="wissensbasis",
    embedding_function=embedding_function
)

'''
Add documents with metadata to the collection
'''


collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)


'''
Query the collection for documents related to "datenbanken"
'''

query = "Wie vermeidet man doppelte Daten in Tabellen?"

results = collection.query(
    query_texts=[query],
    n_results=2
)

print("Suchanfrage:", query)
print("Treffer:")
for doc in results["documents"][0]:
    print("-", doc)



'''
Search with metadata filter
'''

results = collection.query(
    query_texts=["Was ist eine Programmiersprache?"],
    n_results=2,
    where={"topic": "programmiersprachen"}
)

print(results["documents"][0])

'''
Print results with metadata
'''

for doc, meta in zip(
    results["documents"][0],
    results["metadatas"][0]
):
    print(f"[{meta['topic']} | {meta['type']}] {doc}")

