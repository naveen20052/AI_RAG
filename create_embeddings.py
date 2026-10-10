from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# PDF location
pdf_path = "data/sample.pdf"

# Read PDF
reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

# Create chunks
chunk_size = 300

chunks = []

for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(chunks)

# Display results
for i, embedding in enumerate(embeddings):
    print("\n--- CHUNK", i + 1, "---")
    print(chunks[i])

    print("\nEmbedding:")
    print(embedding)

print("\nTotal chunks:", len(chunks))