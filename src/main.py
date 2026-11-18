from pathlib import Path
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
query = "How does embedding-based retrieval work?"
query_tokens = query.lower().split()
# print(query_tokens)

results = []
# tokenizes each chunk
for chunk in chunks:
    chunk_tokens = chunk.lower().split()
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

print(results)
