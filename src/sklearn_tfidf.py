
import string
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Text preprocessing
def preprocess_text(text):
    cleaned_text = text.translate(
        str.maketrans(
            string.punctuation,
            " " * len(string.punctuation)
        )
    )

    stop_words = {"is", "the", "a", "an", "what", "how", "does"}

    cleaned_text = cleaned_text.lower()
    tokens = cleaned_text.split()
    tokens = [word for word in tokens if word not in stop_words]

    return tokens


# 2. Load document
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sample_notes.txt"

with open(DATA_PATH, "r", encoding="utf-8") as file:
    content = file.read()


# 3. Divide document into chunks
paragraphs = content.split("\n\n")

chunks = []

for paragraph in paragraphs:
    cleaned_paragraph = paragraph.strip()

    if cleaned_paragraph:
        chunks.append(cleaned_paragraph)


# 4. Define query
query = "how does RAG retrieve information"


# 5. Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    tokenizer=preprocess_text,
    token_pattern=None,
    lowercase=False
)


# 6. Convert chunks into TF-IDF vectors
chunk_vectors = vectorizer.fit_transform(chunks)


# 7. Convert query into TF-IDF vector
query_vector = vectorizer.transform([query])


# 8. Calculate cosine similarity
similarity_scores = cosine_similarity(
    query_vector,
    chunk_vectors
).flatten()


# 9. Combine chunks with scores
results = list(zip(chunks, similarity_scores))


# 10. Rank chunks
ranked_results = sorted(
    results,
    key=lambda item: item[1],
    reverse=True
)


# 11. Display ranked results
print("\nRanked Results:")

for rank, (chunk, score) in enumerate(ranked_results, start=1):
    print(f"\nRank {rank}:")
    print(f"Score: {score:.3f}")
    print(f"Chunk: {chunk}")


# 12. Display relevant chunks
relevant_chunks = [
    (chunk, score)
    for chunk, score in ranked_results
    if score > 0
]

print("\nRelevant Chunks:")

for chunk, score in relevant_chunks:
    print(f"\nScore: {score:.3f}")
    print(chunk)
