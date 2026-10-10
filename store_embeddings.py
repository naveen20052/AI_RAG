import os
import sys

import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ==========================================
# SETTINGS
# ==========================================

DATA_FOLDER = "data"
CHROMA_FOLDER = "chroma_db"
COLLECTION_NAME = "pdf_documents"

CHUNK_SIZE = 500
OVERLAP = 100


# ==========================================
# CREATE DATA FOLDER
# ==========================================

os.makedirs(
    DATA_FOLDER,
    exist_ok=True
)


# ==========================================
# GET PDF NAME
# ==========================================

if len(sys.argv) > 1:

    pdf_name = sys.argv[1]

else:

    pdf_name = input(
        "Enter PDF filename inside data folder: "
    ).strip()


pdf_name = os.path.basename(
    pdf_name
)


# ==========================================
# CHECK PDF
# ==========================================

if not pdf_name.lower().endswith(".pdf"):

    print(
        "ERROR: Please provide a PDF file."
    )

    sys.exit(1)


pdf_path = os.path.join(
    DATA_FOLDER,
    pdf_name
)


if not os.path.exists(pdf_path):

    print(
        f"ERROR: PDF not found: {pdf_path}"
    )

    sys.exit(1)


print(
    f"\nProcessing: {pdf_name}"
)


# ==========================================
# READ PDF
# ==========================================

reader = PdfReader(
    pdf_path
)


pages = []


for page_number, page in enumerate(
    reader.pages,
    start=1
):

    page_text = page.extract_text()


    if page_text and page_text.strip():

        pages.append({
            "page": page_number,
            "text": page_text
        })


print(
    f"Pages extracted: {len(pages)}"
)


if not pages:

    print(
        "ERROR: No readable text found in this PDF."
    )

    sys.exit(1)


# ==========================================
# CREATE CHUNKS
# ==========================================

chunks = []
metadata = []
ids = []

chunk_number = 0


for page_data in pages:

    page_number = page_data["page"]
    text = page_data["text"]

    start = 0


    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end]


        if chunk.strip():

            chunks.append(
                chunk
            )


            metadata.append({

                "page": page_number,

                "source": pdf_name

            })


            chunk_id = (
                f"{pdf_name}_"
                f"page_{page_number}_"
                f"chunk_{chunk_number}"
            )


            ids.append(
                chunk_id
            )


            chunk_number += 1


        start += (
            CHUNK_SIZE - OVERLAP
        )


print(
    f"Chunks created: {len(chunks)}"
)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print(
    "\nLoading embedding model..."
)


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print(
    "Embedding model loaded."
)


# ==========================================
# CREATE EMBEDDINGS
# ==========================================

print(
    "\nCreating embeddings..."
)


embeddings = model.encode(
    chunks,
    show_progress_bar=True
)


print(
    "Embeddings created."
)


# ==========================================
# CONNECT TO CHROMADB
# ==========================================

print(
    "\nConnecting to ChromaDB..."
)


client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)


collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ==========================================
# REMOVE OLD VERSION OF SAME PDF
# ==========================================

existing = collection.get(
    where={
        "source": pdf_name
    }
)


if existing.get("ids"):

    print(
        f"Removing previous version of "
        f"{pdf_name}..."
    )

    collection.delete(
        ids=existing["ids"]
    )


# ==========================================
# STORE DATA
# ==========================================

collection.add(

    ids=ids,

    documents=chunks,

    embeddings=embeddings.tolist(),

    metadatas=metadata

)


# ==========================================
# RESULT
# ==========================================

print(
    "\n======================================"
)

print(
    "PDF ADDED SUCCESSFULLY"
)

print(
    "======================================"
)

print(
    f"PDF: {pdf_name}"
)

print(
    f"Pages: {len(pages)}"
)

print(
    f"Chunks: {len(chunks)}"
)

print(
    f"Total chunks in knowledge base: "
    f"{collection.count()}"
)

print(
    "======================================"
)