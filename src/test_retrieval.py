
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully!")

# 2. Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

# 3. Load existing collection
collection = client.get_collection(
    name="moderna_safety_guidelines"
)

print("\nCollection:", collection.name)
print("Total stored chunks:", collection.count())

# 4. Enter search question
question = "What are the transport conditions for mRNA products?"

print("\nSearch Question:", question)

# 5. Convert question into embedding
query_embedding = model.encode(
    question,
    normalize_embeddings=True
).tolist()

# 6. Retrieve top 5 relevant chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["documents", "distances", "metadatas"]
)

# 7. Display retrieval results
print("\nRAG RETRIEVAL RESULTS")
print("=" * 50)

for i, document in enumerate(results["documents"][0], start=1):

    distance = results["distances"][0][i - 1]
    metadata = results["metadatas"][0][i - 1]

    print(f"\nResult {i}")
    print("-" * 40)
    print("Chunk Number:", metadata["chunk_number"])
    print("Distance:", round(distance, 4))
    print("\nRetrieved Text:")
    print(document)

print("\n" + "=" * 50)
print("RETRIEVAL TEST COMPLETED!")
