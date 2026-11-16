
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import chromadb
import ollama

from sentence_transformers import SentenceTransformer

# Create output folders
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# Load embedding model and vector database
print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="moderna_safety_guidelines"
)

# Test questions
questions = [
    "What are the transport conditions for the vaccine?",
    "What are the refrigerated storage conditions?",
    "What are the adverse event reporting requirements?"
]

results = []

for question in questions:

    print("\nQuestion:", question)

    # Measure embedding and document retrieval time
    retrieval_start = time.perf_counter()

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    retrieved = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    retrieval_time = time.perf_counter() - retrieval_start

    # Prepare retrieved context
    documents = retrieved["documents"][0]

    context = "\n\n".join(documents)

    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question:
    {question}

    If the answer is not available in the context,
    clearly say that the information is not available.
    """

    # Measure LLM generation time
    llm_start = time.perf_counter()

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

    llm_time = time.perf_counter() - llm_start

    answer = response["message"]["content"]

    # Store actual timings
    results.append({
        "Question": question,
        "Retrieval_Time_Seconds": round(retrieval_time, 4),
        "LLM_Generation_Time_Seconds": round(llm_time, 4),
        "Total_Time_Seconds": round(
            retrieval_time + llm_time, 4
        )
    })

    print("Retrieval time:", round(retrieval_time, 4), "seconds")
    print("LLM generation time:", round(llm_time, 4), "seconds")
    print("Answer:", answer[:200])

# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/reports/latency_comparison.csv",
    index=False
)

# Calculate average timings
average_retrieval = results_df[
    "Retrieval_Time_Seconds"
].mean()

average_llm = results_df[
    "LLM_Generation_Time_Seconds"
].mean()

# Create latency comparison chart
plt.figure(figsize=(8, 5))

plt.bar(
    ["RAG Retrieval", "LLM Generation"],
    [average_retrieval, average_llm]
)

plt.ylabel("Average Time (Seconds)")
plt.title("RAG vs LLM Latency Comparison")
plt.tight_layout()

plt.savefig(
    "outputs/figures/latency_comparison.png",
    dpi=300
)

plt.close()

print("\n" + "=" * 45)
print("LATENCY BENCHMARK COMPLETED")
print("=" * 45)

print("Average retrieval time:", round(average_retrieval, 4), "seconds")
print("Average LLM generation time:", round(average_llm, 4), "seconds")

print("\nSaved files:")
print("outputs/reports/latency_comparison.csv")
print("outputs/figures/latency_comparison.png")
