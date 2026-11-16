
import os
import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer

# Configuration
INPUT_FILE = "data/mrna_safety_guidelines.txt"
DB_PATH = "chroma_db"
COLLECTION_NAME = "moderna_safety_guidelines"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30

# 1. Load guidelines
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    text = file.read()

print("Guidelines loaded successfully!")
print("Characters:", len(text))

# 2. Tokenize text
encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode(text)

print("Total tokens:", len(tokens))

# 3. Create overlapping chunks
chunks = []

start = 0

while start < len(tokens):
    end = min(start + CHUNK_SIZE, len(tokens))

    chunk_tokens = tokens[start:end]
    chunk_text = encoding.decode(chunk_tokens)

    if chunk_text.strip():
        chunks.append(chunk_text)

    if end == len(tokens):
        break

    start = end - CHUNK_OVERLAP

print("\nTotal chunks created:", len(chunks))

# 4. Load embedding model
print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")

# 5. Generate embeddings
embeddings = model.encode(
    chunks,
    show_progress_bar=True,
    normalize_embeddings=True
)

# 6. Create persistent ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)

# Rebuild collection when rerunning this script
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "Moderna safety guidelines"}
)

# 7. Store documents and embeddings
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist(),
    metadatas=[
        {
            "source": "mrna_safety_guidelines.txt",
            "chunk_number": i
        }
        for i in range(len(chunks))
    ]
)

# 8. Verify database
print("\nDATABASE BUILD COMPLETED")
print("Collection:", COLLECTION_NAME)
print("Stored chunks:", collection.count())
print("Database location:", DB_PATH)

print("\nRAG knowledge base created successfully!")
