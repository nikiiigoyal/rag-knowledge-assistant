
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

# 1. Load document

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sample_notes.txt"

with open(DATA_PATH, "r", encoding="utf-8") as file:
    content = file.read()


# 2. Divide document into chunks

paragraphs = content.split("\n\n")

chunks = []

for paragraph in paragraphs:
    cleaned_paragraph = paragraph.strip()

    if cleaned_paragraph:
        chunks.append(cleaned_paragraph)

print("Total chunks:", len(chunks))


# 3. Load embedding model

model = SentenceTransformer("all-MiniLM-L6-v2")


# 4. Generate embeddings for all document chunks

chunk_embeddings = model.encode(chunks)

print("Total embeddings:", len(chunk_embeddings))
print("Dimensions per embedding:", len(chunk_embeddings[0]))


# 5. Define user query

query = "Who invented Python?"
top_k = 3


# 6. Create a retrieval function

def retrieve_context(query, top_k=5):

    query_embedding = model.encode(query)

    similarity_scores = cosine_similarity(
        [query_embedding],
        chunk_embeddings
    ).flatten()

    results = list(zip(chunks, similarity_scores))

    ranked_results = sorted(
        results,
        key=lambda item: item[1],
        reverse=True
    )

    # Filter chunks using similarity threshold
    similarity_threshold = 0.30

    top_results = [
        (chunk, score)
        for chunk, score in ranked_results
        if score >= similarity_threshold
    ][:top_k]

    return top_results


# 7. Test retrieval

retrieved_results = retrieve_context(query, top_k=top_k)

if not retrieved_results:
    print("No sufficiently relevant information found.")
    exit()

print("\nTop Retrieved Chunks:")

for rank, (chunk, score) in enumerate(retrieved_results, start=1):
    print(f"\nRank {rank}")
    print(f"Score: {score:.3f}")
    print(f"Chunk: {chunk}")

context = "\n\n".join(
    chunk for chunk, score in retrieved_results
)


# 8. Create RAG prompt

prompt = f"""
Answer the question using only the provided context.

If the answer is not available in the context,
say that you don't have enough information.

Context:
{context}

Question:
{query}

Answer:
"""

print("\nRAG Prompt:")
print(prompt)


# 9. Send prompt to Ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# 10. Display generated answer

answer = response["message"]["content"]

print("\nGenerated Answer:")
print(answer)
