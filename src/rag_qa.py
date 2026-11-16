
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# 1. Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="moderna_safety_guidelines"
)

# 3. Retrieve relevant information
def retrieve_information(question, top_k=5):

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )

    return results


# 4. Generate answer using Llama
def generate_answer(question):

    results = retrieve_information(question)

    documents = results["documents"][0]
    distances = results["distances"][0]

    # Rank retrieved chunks using keyword matching
    keywords = [
        word.lower()
        for word in question.split()
        if len(word) > 3
    ]

    ranked_documents = []

    for i, document in enumerate(documents):

        text = document.lower()

        score = sum(
            text.count(keyword)
            for keyword in keywords
        )

        if "transport" in question.lower():
            if "1.3 transport conditions" in text:
                score += 10

        ranked_documents.append(
            (score, distances[i], document)
        )

    ranked_documents.sort(
        key=lambda item: item[0],
        reverse=True
    )

    # Select top relevant passages
    context = "\n\n".join(
        item[2] for item in ranked_documents[:3]
    )

    # 5. Prepare prompt
    prompt = f"""
You are an educational research assistant.

Answer the question using ONLY the supplied project reference context.

Rules:
- Do not invent facts.
- If the context does not contain the answer, say so.
- Keep the answer simple and clear.
- Do not provide real medical, vaccine dosage, or treatment advice.
- Treat the supplied document as educational project material,
  not official clinical or regulatory guidance.

REFERENCE CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    print("\nGenerating answer using Llama 3.2...")

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1
        }
    )

    print("\nRAG GENERATED ANSWER")
    print("=" * 50)
    print(response["message"]["content"])

    print("\nSOURCE")
    print("mrna_safety_guidelines.txt")


# 6. Run application
if __name__ == "__main__":

    print("\nMODERNA mRNA RAG QUESTION-ANSWERING SYSTEM")
    print("-" * 50)

    question = input("\nEnter your question: ")

    if question.strip():
        generate_answer(question)
    else:
        print("Please enter a valid question.")
