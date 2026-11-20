
import string
import math
from pathlib import Path


# Text preprocessing
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


# Calculate cosine similarity between two vectors
def cosine_similarity(A, B):

    if len(A) != len(B):
        raise ValueError("Vectors must have the same dimensions")

    dot_product = 0

    for i in range(len(A)):
        dot_product += A[i] * B[i]

    magnitude_A = math.sqrt(sum(x ** 2 for x in A))
    magnitude_B = math.sqrt(sum(x ** 2 for x in B))

    if magnitude_A == 0 or magnitude_B == 0:
        return 0.0

    similarity = dot_product / (magnitude_A * magnitude_B)

    return similarity


# 1. Parse the document
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

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}:")


# 3. Tokenize query
query = "what is cosine similarity"

query_tokens = list(
    dict.fromkeys(preprocess_text(query))
)

print("\nQuery tokens:", query_tokens)

results = []


# 4. Calculate total number of chunks
N = len(chunks)


# 5. Collect all unique words across all chunks
all_words = set()

for chunk in chunks:
    chunk_tokens = preprocess_text(chunk)
    all_words.update(chunk_tokens)

# Create vocabulary after collecting all words
vocabulary = sorted(all_words)

print("\nTotal unique words:", len(all_words))
print("Vocabulary:", vocabulary)


# 6. Calculate Document Frequency (DF)
document_frequency = {}

for word in all_words:
    df = 0

    for chunk in chunks:
        chunk_tokens = preprocess_text(chunk)

        if word in chunk_tokens:
            df += 1

    document_frequency[word] = df

print("\nDocument Frequency:")

for word, df in document_frequency.items():
    print(f"{word}: {df}")


# 7. Calculate Inverse Document Frequency (IDF)
inverse_document_frequency = {}

for word, df in document_frequency.items():
    idf = math.log((N + 1) / (df + 1)) + 1

    inverse_document_frequency[word] = idf

print("\nInverse Document Frequency:")

for word, idf in inverse_document_frequency.items():
    print(f"{word}: {idf:.2f}")


# 8. Calculate TF and TF-IDF for each chunk
for chunk in chunks:

    chunk_tokens = preprocess_text(chunk)

    # Handle empty chunks
    if not chunk_tokens:
        continue

    unique_words = set(chunk_tokens)
    total_words = len(chunk_tokens)

    # Calculate TF-IDF for each word
    for word in unique_words:

        frequency = chunk_tokens.count(word)
        term_freq = frequency / total_words

        tf_idf = term_freq * inverse_document_frequency[word]

        print(
            f"{word}: TF={term_freq:.3f}, "
            f"TF-IDF={tf_idf:.3f}"
        )

    # 9. Calculate TF-IDF score for the query
    tf_idf_score = 0

    for word in query_tokens:

        if word in chunk_tokens:

            frequency = chunk_tokens.count(word)
            term_freq = frequency / total_words

            idf = inverse_document_frequency[word]

            word_score = term_freq * idf

            tf_idf_score += word_score

            print(f"{word}: TF-IDF = {word_score:.3f}")

    # Store the chunk and its score
    print(f"\nChunk: {chunk[:50]}...")
    print(f"Total TF-IDF Score: {tf_idf_score:.3f}")

    results.append((chunk, tf_idf_score))


# 10. Rank chunks based on TF-IDF score
ranked_results = sorted(
    results,
    key=lambda item: item[1],
    reverse=True
)

for rank, (chunk, score) in enumerate(ranked_results, start=1):

    print(f"\nRank {rank}:")
    print(f"Score: {score:.3f}")
    print(f"Chunk: {chunk[:100]}...")


# 11. Remove chunks with zero relevance score
relevant_chunks = [
    item for item in ranked_results
    if item[1] > 0
]

print("\nRelevant Chunks:")

for chunk, score in relevant_chunks:
    print(f"Score: {score:.3f}")
    print(chunk)
