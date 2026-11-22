from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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
query = "How does RAG retrieve information?"


# 6. Generate query embedding
query_embedding = model.encode(query)
print("Query embedding dimensions:", len(query_embedding))

# 7. Calculate similarity scores

similarity_scores = cosine_similarity(
    [query_embedding],
    chunk_embeddings
).flatten()

print("Similarity scores:", similarity_scores)

# 8. Combine chunks with similarity scores
results = list(zip(chunks, similarity_scores))

# 9. Rank chunks by similarity score

ranked_results = sorted(
    results,
    key=lambda item: item[1],
    reverse=True
)
# 10. Display top 3 results

print("\nTop 3 Relevant Chunks:")

for rank, (chunk, score) in enumerate(
    ranked_results[:3], start=1
):
    print(f"\nRank {rank}")
    print(f"Score: {score:.3f}")
    print(f"Chunk: {chunk}")
