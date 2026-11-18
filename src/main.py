import string
from pathlib import Path

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
    tokens = list(dict.fromkeys(tokens))

    return tokens
# parsing whole doc

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "sample_notes.txt"

with open(DATA_PATH, "r" , encoding="utf-8") as file:
    content = file.read()
    # print(content)
paragraphs = content.split("\n\n")

# dividing into chunks

chunks = []

for paragraph in paragraphs:
    cleaned_paragraph = paragraph.strip()
    if cleaned_paragraph:
        chunks.append(cleaned_paragraph)
# print(f"Total chunks: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}:")
    # print(chunk)

# tokenizes query
query = "what is cosine similarity"
query_tokens = preprocess_text(query)

print(query_tokens)

results = []
# tokenizes each chunk
for chunk in chunks:
    chunk_tokens = preprocess_text(chunk)
# find mathcing words between query and chunk
    matching_words = []

    for word in query_tokens:
        if word in chunk_tokens:
            matching_words.append(word)
   
   # calculate relevance score for each chunk 
    score = len(matching_words) 
    print(f"Chunk: {chunk[:50]}...")
    print("Matching words:", ", ".join(matching_words))
    print(f"Score: {score}")

    results.append((chunk, score))

# print(results)

# rank chunks based on relevance score
ranked_results = sorted(results, key=lambda item: item[1], reverse=True )
# print(ranked_results)

for rank, (chunk, score) in enumerate(ranked_results, start=1):
    print(f"\nRank {rank}:")
    print(f"Score: {score}")
    print(f"Chunk: {chunk[:100]}..." )

# removing chunks with zero relevance score
relevant_chunks = [item for item in ranked_results
                   if item[1] > 0]
print("\nRelevant Chunks:")
for chunk, score in relevant_chunks:
    print(f"Score: {score}")
    print(chunk)

