import os
import re

import chromadb
import fitz
import pytesseract

from flask import (
    Flask,
    render_template,
    request,
    session,
    send_from_directory
)

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from PIL import Image

from rank_bm25 import BM25Okapi

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from werkzeug.utils import secure_filename

from database import (
    init_database,
    save_chat,
    get_chat_history,
    delete_pdf_history,
    delete_all_history,
    save_summary,
    get_summary,
    delete_pdf_summary,
    delete_all_summaries
)


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()


# ==================================================
# FLASK APP
# ==================================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "change-this-secret-key"
)


# ==================================================
# SETTINGS
# ==================================================

DATA_FOLDER = "data"

CHROMA_FOLDER = "chroma_db"

COLLECTION_NAME = "pdf_documents"

CHUNK_SIZE = 500

OVERLAP = 100

RETRIEVAL_K = 8

BM25_K = 8

FINAL_K = 5

RRF_K = 60

RERANK_MIN_SCORE = float(
    os.getenv("RERANK_MIN_SCORE", "0.0")
)

OCR_SCALE = 2.5

ALL_PDF_HISTORY_KEY = "__ALL_PDFS__"


# ==================================================
# CREATE FOLDERS
# ==================================================

os.makedirs(
    DATA_FOLDER,
    exist_ok=True
)

os.makedirs(
    CHROMA_FOLDER,
    exist_ok=True
)


# ==================================================
# SQLITE
# ==================================================

init_database()

print(
    "SQLite database initialized."
)


# ==================================================
# TESSERACT OCR
# ==================================================

TESSERACT_PATHS = [

    r"C:\Program Files\Tesseract-OCR\tesseract.exe",

    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

]


OCR_AVAILABLE = False


for tesseract_path in TESSERACT_PATHS:

    if os.path.exists(
        tesseract_path
    ):

        pytesseract.pytesseract.tesseract_cmd = (
            tesseract_path
        )

        print(
            f"Tesseract found: {tesseract_path}"
        )

        break


try:

    pytesseract.get_tesseract_version()

    OCR_AVAILABLE = True

    print(
        "OCR is available."
    )

except Exception:

    print(
        "OCR is not available."
    )

    print(
        "Normal text PDFs will still work."
    )


# ==================================================
# GROQ
# ==================================================

api_key = os.getenv(
    "GROQ_API_KEY"
)


if not api_key:

    raise ValueError(
        "GROQ_API_KEY not found in .env file"
    )


groq_client = Groq(

    api_key=api_key,

    timeout=60,

    max_retries=1

)


# ==================================================
# EMBEDDING MODEL
# ==================================================

print(
    "Loading embedding model..."
)


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print(
    "Embedding model loaded."
)


# ==================================================
# RERANKER
# ==================================================

print(
    "Loading reranker model..."
)


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


print(
    "Reranker model loaded."
)


# ==================================================
# CHROMADB
# ==================================================

client = chromadb.PersistentClient(

    path=CHROMA_FOLDER

)


collection = client.get_or_create_collection(

    name=COLLECTION_NAME

)


print(
    "ChromaDB connected."
)


print(
    f"Stored chunks: {collection.count()}"
)


# ==================================================
# BM25 HYBRID RETRIEVAL
# ==================================================

bm25 = None
bm25_documents = []
bm25_metadatas = []
bm25_ids = []


def tokenize_text(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


def rebuild_bm25():

    global bm25
    global bm25_documents
    global bm25_metadatas
    global bm25_ids

    print("Rebuilding BM25 index...")

    data = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    bm25_documents = data.get(
        "documents",
        []
    )

    bm25_metadatas = data.get(
        "metadatas",
        []
    )

    bm25_ids = data.get(
        "ids",
        []
    )

    if not bm25_documents:

        bm25 = None

        print("BM25 index is empty.")

        return

    tokenized_documents = [
        tokenize_text(document)
        for document in bm25_documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    print(
        f"BM25 index ready: {len(bm25_documents)} chunks"
    )


def bm25_search(question, top_k=BM25_K, source=None):

    if bm25 is None:
        return []

    query_tokens = tokenize_text(question)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    candidates = []

    for index, score in enumerate(scores):

        metadata = bm25_metadatas[index]

        if source is not None and metadata.get("source") != source:
            continue

        candidates.append({
            "id": bm25_ids[index],
            "document": bm25_documents[index],
            "metadata": metadata,
            "score": float(score)
        })

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return candidates[:top_k]


def semantic_search(question_embedding, source=None, top_k=RETRIEVAL_K):

    if collection.count() == 0:
        return []

    if source is not None:

        source_data = collection.get(
            where={"source": source}
        )

        if not source_data.get("ids", []):
            return []

        top_k = min(
            top_k,
            len(source_data["ids"])
        )

    else:
        top_k = min(
            top_k,
            collection.count()
        )

    query_kwargs = {
        "query_embeddings": [question_embedding],
        "n_results": top_k,
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    if source is not None:
        query_kwargs["where"] = {"source": source}

    results = collection.query(**query_kwargs)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    candidates = []

    for index, document in enumerate(documents):

        candidates.append({
            "id": ids[index],
            "document": document,
            "metadata": metadatas[index],
            "distance": distances[index]
        })

    return candidates


def hybrid_search(question, question_embedding, source=None):

    semantic_results = semantic_search(
        question_embedding,
        source=source,
        top_k=RETRIEVAL_K
    )

    bm25_results = bm25_search(
        question,
        top_k=BM25_K,
        source=source
    )

    print(
        f"Semantic candidates: {len(semantic_results)}"
    )
    print(
        f"BM25 candidates: {len(bm25_results)}"
    )

    combined = {}

    for rank, item in enumerate(semantic_results, start=1):

        key = item["id"]

        combined.setdefault(
            key,
            {
                "document": item["document"],
                "metadata": item["metadata"],
                "rrf_score": 0.0
            }
        )

        combined[key]["rrf_score"] += (
            1 / (RRF_K + rank)
        )

    for rank, item in enumerate(bm25_results, start=1):

        key = item["id"]

        combined.setdefault(
            key,
            {
                "document": item["document"],
                "metadata": item["metadata"],
                "rrf_score": 0.0
            }
        )

        combined[key]["rrf_score"] += (
            1 / (RRF_K + rank)
        )

    candidates = list(combined.values())

    candidates.sort(
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    print(
        f"Hybrid candidates: {len(candidates)}"
    )

    return candidates


# Build BM25 from the persistent ChromaDB corpus at startup.
rebuild_bm25()


# ==================================================
# PDF HELPERS
# ==================================================

def allowed_file(
    filename
):

    return (

        "." in filename

        and

        filename.rsplit(
            ".",
            1
        )[1].lower() == "pdf"

    )


def get_uploaded_pdfs():

    return sorted([

        filename

        for filename in os.listdir(
            DATA_FOLDER
        )

        if filename.lower().endswith(
            ".pdf"
        )

    ])


def get_pdf_count():

    return len(
        get_uploaded_pdfs()
    )


# ==================================================
# OCR PAGE
# ==================================================

def ocr_pdf_page(
    pdf_path,
    page_number
):

    if not OCR_AVAILABLE:

        raise RuntimeError(
            "Tesseract OCR is not installed."
        )


    pdf = fitz.open(
        pdf_path
    )


    try:

        page = pdf[
            page_number - 1
        ]


        matrix = fitz.Matrix(
            OCR_SCALE,
            OCR_SCALE
        )


        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )


        image = Image.frombytes(

            "RGB",

            (
                pix.width,
                pix.height
            ),

            pix.samples

        )


        text = pytesseract.image_to_string(
            image
        )


        return text.strip()


    finally:

        pdf.close()


# ==================================================
# EXTRACT PDF PAGES
# ==================================================

def extract_pdf_pages(
    pdf_path
):

    reader = PdfReader(
        pdf_path
    )


    pages = []


    for page_number, page in enumerate(

        reader.pages,

        start=1

    ):

        text = page.extract_text()


        if text:

            text = text.strip()

        else:

            text = ""


        # ------------------------------------------
        # NORMAL TEXT PDF
        # ------------------------------------------

        if len(text.split()) >= 5:

            print(
                f"Page {page_number}: PyPDF"
            )


            pages.append({

                "page": page_number,

                "text": text,

                "method": "PyPDF"

            })


        # ------------------------------------------
        # SCANNED PAGE
        # ------------------------------------------

        else:

            print(
                f"Page {page_number}: OCR fallback"
            )


            if not OCR_AVAILABLE:

                continue


            ocr_text = ocr_pdf_page(

                pdf_path,

                page_number

            )


            if ocr_text:

                pages.append({

                    "page": page_number,

                    "text": ocr_text,

                    "method": "OCR"

                })


    return pages


# ==================================================
# INDEX PDF
# ==================================================

def index_pdf(
    pdf_path,
    pdf_name
):

    print(
        "\n======================================"
    )

    print(
        f"Processing PDF: {pdf_name}"
    )

    print(
        "======================================"
    )


    # ------------------------------------------
    # EXTRACT
    # ------------------------------------------

    pages = extract_pdf_pages(
        pdf_path
    )


    if not pages:

        raise ValueError(
            "No readable text found in this PDF."
        )


    print(
        f"Readable pages: {len(pages)}"
    )


    # ------------------------------------------
    # CHUNK
    # ------------------------------------------

    chunks = []

    metadatas = []

    ids = []

    chunk_number = 0


    for page_data in pages:

        page_number = page_data[
            "page"
        ]

        text = page_data[
            "text"
        ]

        method = page_data[
            "method"
        ]


        start = 0


        while start < len(text):

            end = start + CHUNK_SIZE

            chunk = text[
                start:end
            ]


            if chunk.strip():

                chunks.append(
                    chunk
                )


                metadatas.append({

                    "source":
                        pdf_name,

                    "page":
                        page_number,

                    "method":
                        method

                })


                ids.append(

                    f"{pdf_name}_"
                    f"page_{page_number}_"
                    f"chunk_{chunk_number}"

                )


                chunk_number += 1


            start += (
                CHUNK_SIZE - OVERLAP
            )


    print(
        f"Chunks created: {len(chunks)}"
    )


    if not chunks:

        raise ValueError(
            "No chunks were created."
        )


    # ------------------------------------------
    # EMBEDDINGS
    # ------------------------------------------

    print(
        "Creating embeddings..."
    )


    embeddings = embedding_model.encode(

        chunks,

        show_progress_bar=True

    )


    print(
        "Embeddings created."
    )


    # ------------------------------------------
    # DELETE OLD VERSION
    # ------------------------------------------

    try:

        old_data = collection.get(

            where={
                "source":
                    pdf_name
            }

        )


        old_ids = old_data.get(
            "ids",
            []
        )


        if old_ids:

            collection.delete(
                ids=old_ids
            )


    except Exception as e:

        print(
            "Old version cleanup skipped:",
            str(e)
        )


    # ------------------------------------------
    # STORE
    # ------------------------------------------

    collection.add(

        ids=ids,

        documents=chunks,

        embeddings=embeddings.tolist(),

        metadatas=metadatas

    )


    rebuild_bm25()

    print(
        "PDF indexed successfully."
    )


    print(
        f"Total chunks: {collection.count()}"
    )


# ==================================================
# RERANK
# ==================================================

def rerank_documents(

    question,

    documents,

    metadatas

):

    if not documents:

        return [], []


    pairs = [

        [
            question,
            document
        ]

        for document in documents

    ]


    scores = reranker.predict(
        pairs
    )


    ranked = []


    for (

        score,

        document,

        metadata

    ) in zip(

        scores,

        documents,

        metadatas

    ):


        ranked.append({

            "score":
                float(score),

            "document":
                document,

            "metadata":
                metadata

        })


    ranked.sort(

        key=lambda item:
            item["score"],

        reverse=True

    )


    print("\nReranker scores:")

    for item in ranked:

        print(
            f"Score={item['score']:.4f} "
            f"Source={item['metadata'].get('source')} "
            f"Page={item['metadata'].get('page')}"
        )


    # ------------------------------------------
    # RELEVANCE GATE
    # ------------------------------------------

    relevant = [

        item

        for item in ranked

        if item["score"] >= RERANK_MIN_SCORE

    ]


    if not relevant:

        print(
            "No chunks passed the relevance threshold "
            f"({RERANK_MIN_SCORE})."
        )

        return [], []


    relevant = relevant[:FINAL_K]


    print(
        f"Relevant chunks after threshold: "
        f"{len(relevant)}"
    )


    final_documents = [

        item["document"]

        for item in relevant

    ]


    final_metadatas = [

        item["metadata"]

        for item in relevant

    ]


    return (

        final_documents,

        final_metadatas

    )


# ==================================================
# GENERATE SUMMARY
# ==================================================

def generate_pdf_summary(
    pdf_name
):

    print(
        f"\nGenerating summary for: {pdf_name}"
    )


    result = collection.get(

        where={
            "source":
                pdf_name
        },

        include=[
            "documents"
        ]

    )


    documents = result.get(
        "documents",
        []
    )


    if not documents:

        raise ValueError(
            "No indexed content found for this PDF."
        )


    print(
        f"Summary source chunks: {len(documents)}"
    )


    # ------------------------------------------
    # PARTIAL SUMMARIES
    # ------------------------------------------

    batch_size = 6

    partial_summaries = []


    for start in range(

        0,

        len(documents),

        batch_size

    ):


        batch = documents[
            start:start + batch_size
        ]


        batch_text = "\n\n".join(
            batch
        )


        prompt = f"""
You are summarizing a PDF document.

Summarize ONLY the information contained
in the provided document content.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Keep important concepts and key facts.
4. Remove unnecessary repetition.
5. Use clear and simple language.
6. Use headings and bullet points when useful.

DOCUMENT CONTENT:

{batch_text}
"""


        response = (

            groq_client
            .chat
            .completions
            .create(

                model=
                    "openai/gpt-oss-20b",

                messages=[

                    {

                        "role":
                            "system",

                        "content":
                            (
                                "Summarize only "
                                "the supplied "
                                "document content."
                            )

                    },

                    {

                        "role":
                            "user",

                        "content":
                            prompt

                    }

                ],

                temperature=0

            )

        )


        partial_summary = (

            response
            .choices[0]
            .message
            .content

        )


        partial_summaries.append(
            partial_summary
        )


        print(
            f"Summary batch "
            f"{(start // batch_size) + 1} completed."
        )


    # ------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------

    combined = "\n\n".join(
        partial_summaries
    )


    final_prompt = f"""
Create a final summary of the PDF using ONLY
the intermediate summaries below.

Rules:

1. Do not add outside knowledge.
2. Do not invent information.
3. Combine repeated information.
4. Keep important concepts and facts.
5. Make it easy to read.
6. Use headings and bullet points.
7. Keep the final summary reasonably concise.

INTERMEDIATE SUMMARIES:

{combined}

FINAL SUMMARY:
"""


    response = (

        groq_client
        .chat
        .completions
        .create(

            model=
                "openai/gpt-oss-20b",

            messages=[

                {

                    "role":
                        "system",

                    "content":
                        (
                            "Create a final "
                            "summary only "
                            "from the supplied "
                            "intermediate summaries."
                        )

                },

                {

                    "role":
                        "user",

                    "content":
                        final_prompt

                }

            ],

            temperature=0

        )

    )


    summary = (

        response
        .choices[0]
        .message
        .content

    )


    return summary


# ==================================================
# HOME
# ==================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    answer = None

    message = None

    question = ""

    source_pages = []

    search_mode = "current"


    if request.method == "POST":


        question = request.form.get(

            "question",

            ""

        ).strip()


        search_mode = request.form.get(

            "search_mode",

            "current"

        )


        if search_mode not in [

            "current",

            "all"

        ]:

            search_mode = "current"


        active_pdf = session.get(

            "active_pdf"

        )


        if not question:


            message = (

                "Please enter a question."

            )


        elif (

            search_mode == "current"

            and

            not active_pdf

        ):


            message = (

                "Please upload or select "
                "a PDF first."

            )


        elif collection.count() == 0:


            message = (

                "Please upload at least "
                "one PDF first."

            )


        else:


            try:


                print(
                    "\n======================================"
                )


                print(
                    f"Question: {question}"
                )


                print(
                    f"Search mode: {search_mode}"
                )


                if active_pdf:

                    print(
                        f"Active PDF: {active_pdf}"
                    )


                # ==================================
                # STEP 1
                # EMBEDDING
                # ==================================

                print(
                    "[1/5] Creating question embedding..."
                )


                question_embedding = (

                    embedding_model
                    .encode(question)
                    .tolist()

                )


                # ==================================
                # STEP 2
                # HYBRID RETRIEVAL
                # ==================================

                print(
                    "[2/5] Running hybrid retrieval..."
                )

                active_source = None

                if search_mode == "current":
                    active_source = active_pdf

                hybrid_candidates = hybrid_search(
                    question,
                    question_embedding,
                    source=active_source
                )

                if not hybrid_candidates:
                    raise ValueError(
                        "No relevant documents found."
                    )

                documents = [
                    item["document"]
                    for item in hybrid_candidates
                ]

                metadatas = [
                    item["metadata"]
                    for item in hybrid_candidates
                ]

                print(
                    f"Hybrid candidates returned: {len(documents)}"
                )


                # ==================================
                # STEP 3
                # RERANK
                # ==================================

                print(
                    "[3/5] Reranking chunks..."
                )


                (

                    documents,

                    metadatas

                ) = rerank_documents(

                    question,

                    documents,

                    metadatas

                )


                print(

                    f"Final chunks after relevance gate: "
                    f"{len(documents)}"

                )


                # ==================================
                # BUILD CONTEXT
                # ==================================

                context_parts = []


                for (

                    document,

                    metadata

                ) in zip(

                    documents,

                    metadatas

                ):


                    source = metadata.get(

                        "source",

                        "Unknown PDF"

                    )


                    page = metadata.get(

                        "page",

                        "Unknown"

                    )


                    method = metadata.get(

                        "method",

                        "Unknown"

                    )


                    context_parts.append(

                        f"""
PDF: {source}

Page: {page}

Extraction method: {method}

Content:
{document}
"""

                    )


                    source_info = {

                        "filename":
                            source,

                        "page":
                            page,

                        "label":
                            (
                                f"{source} "
                                f"- Page {page}"
                            )

                    }


                    duplicate = any(

                        item["filename"] == source

                        and

                        item["page"] == page

                        for item in source_pages

                    )


                    if not duplicate:

                        source_pages.append(
                            source_info
                        )


                if not context_parts:


                    answer = (

                        "I could not find the "
                        "answer in the uploaded PDFs."

                    )


                else:


                    context = "\n".join(

                        context_parts

                    )


                    # ==================================
                    # MODE
                    # ==================================

                    if search_mode == "current":

                        mode_instruction = f"""

CURRENT PDF MODE

Use only:

{active_pdf}

"""


                    else:

                        mode_instruction = """

ALL PDF MODE

You may use information from
different uploaded PDFs.

"""


                    # ==================================
                    # PROMPT
                    # ==================================

                    prompt = f"""
You are a Retrieval-Augmented Generation
(RAG) document question-answering assistant.

{mode_instruction}

Answer the user's question using ONLY the
retrieved PDF context below.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. Use only the supplied PDF context.
4. Answer directly and clearly.
5. In ALL PDF mode, you may combine information
   from multiple PDFs.
6. Mention the source PDF when useful.
7. If the answer cannot be found in the context,
   say exactly:

"I could not find the answer in the uploaded PDFs."

PDF CONTEXT:

{context}

USER QUESTION:

{question}
"""


                    # ==================================
                    # STEP 4
                    # GROQ
                    # ==================================

                    print(
                        "[4/5] Asking Groq..."
                    )


                    response = (

                        groq_client
                        .chat
                        .completions
                        .create(

                            model=
                                "openai/gpt-oss-20b",

                            messages=[

                                {

                                    "role":
                                        "system",

                                    "content":
                                        (
                                            "Answer only "
                                            "from the supplied "
                                            "PDF context."
                                        )

                                },

                                {

                                    "role":
                                        "user",

                                    "content":
                                        prompt

                                }

                            ],

                            temperature=0

                        )

                    )


                    answer = (

                        response
                        .choices[0]
                        .message
                        .content

                    )


                    # ==================================
                    # STEP 5
                    # SAVE CHAT
                    # ==================================

                    print(
                        "[5/5] Saving chat..."
                    )


                    history_pdf = (

                        active_pdf

                        if search_mode == "current"

                        else ALL_PDF_HISTORY_KEY

                    )


                    save_chat(

                        pdf_name=
                            history_pdf,

                        question=
                            question,

                        answer=
                            answer,

                        search_mode=
                            search_mode,

                        sources=
                            source_pages

                    )


                    print(
                        "Answer generated."
                    )


            except Exception as e:


                print(
                    "\nERROR OCCURRED"
                )


                print(
                    "Type:",
                    type(e).__name__
                )


                print(
                    "Message:",
                    str(e)
                )


                message = (

                    f"Error: {str(e)}"

                )


    active_pdf = session.get(
        "active_pdf"
    )


    history_key = (

        active_pdf

        if search_mode == "current"

        else ALL_PDF_HISTORY_KEY

    )


    chat_history = (

        get_chat_history(
            history_key
        )

        if history_key

        else []

    )


    stored_summary = (

        get_summary(active_pdf)

        if active_pdf

        else None

    )


    return render_template(

        "index.html",

        question=question,

        answer=answer,

        message=message,

        source_pages=source_pages,

        pdf_count=get_pdf_count(),

        uploaded_pdfs=get_uploaded_pdfs(),

        active_pdf=active_pdf,

        search_mode=search_mode,

        chat_history=chat_history,

        summary=(
            stored_summary["summary"]
            if stored_summary
            else None
        ),

        summary_updated_at=(
            stored_summary["updated_at"]
            if stored_summary
            else None
        )

    )


# ==================================================
# UPLOAD PDF
# ==================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_pdf():

    if "pdf" not in request.files:

        return render_template(

            "index.html",

            message="No PDF selected.",

            pdf_count=get_pdf_count(),

            uploaded_pdfs=get_uploaded_pdfs(),

            active_pdf=session.get(
                "active_pdf"
            ),

            search_mode="current",

            chat_history=[]

        )


    file = request.files["pdf"]


    if file.filename == "":

        return render_template(

            "index.html",

            message="Please select a PDF.",

            pdf_count=get_pdf_count(),

            uploaded_pdfs=get_uploaded_pdfs(),

            active_pdf=session.get(
                "active_pdf"
            ),

            search_mode="current",

            chat_history=[]

        )


    if not allowed_file(
        file.filename
    ):

        return render_template(

            "index.html",

            message="Only PDF files are allowed.",

            pdf_count=get_pdf_count(),

            uploaded_pdfs=get_uploaded_pdfs(),

            active_pdf=session.get(
                "active_pdf"
            ),

            search_mode="current",

            chat_history=[]

        )


    filename = secure_filename(
        file.filename
    )


    pdf_path = os.path.join(

        DATA_FOLDER,

        filename

    )


    file.save(
        pdf_path
    )


    try:

        # ------------------------------------------
        # INDEX PDF
        # ------------------------------------------

        index_pdf(

            pdf_path,

            filename

        )


        # ------------------------------------------
        # MAKE ACTIVE
        # ------------------------------------------

        session[
            "active_pdf"
        ] = filename


        message = (

            f"{filename} uploaded successfully. "
            "It is now the active PDF."

        )


    except Exception as e:

        print(
            "\nUPLOAD ERROR:"
        )

        print(
            type(e).__name__
        )

        print(
            str(e)
        )


        message = (

            f"PDF processing failed: "
            f"{str(e)}"

        )


    return render_template(

        "index.html",

        message=message,

        pdf_count=get_pdf_count(),

        uploaded_pdfs=get_uploaded_pdfs(),

        active_pdf=session.get(
            "active_pdf"
        ),

        search_mode="current",

        chat_history=(
            get_chat_history(
                session.get(
                    "active_pdf"
                )
            )
            if session.get(
                "active_pdf"
            )
            else []
        ),

        summary=(
            get_summary(
                session.get(
                    "active_pdf"
                )
            )["summary"]
            if (
                session.get(
                    "active_pdf"
                )
                and
                get_summary(
                    session.get(
                        "active_pdf"
                    )
                )
            )
            else None
        )

    )


# ==================================================
# SELECT PDF
# ==================================================

@app.route(
    "/select/<path:filename>",
    methods=["POST"]
)
def select_pdf(filename):

    filename = os.path.basename(
        filename
    )


    pdf_path = os.path.join(

        DATA_FOLDER,

        filename

    )


    if not os.path.exists(
        pdf_path
    ):

        return render_template(

            "index.html",

            message="PDF not found.",

            pdf_count=get_pdf_count(),

            uploaded_pdfs=get_uploaded_pdfs(),

            active_pdf=None,

            search_mode="current",

            chat_history=[],

            summary=None

        )


    session[
        "active_pdf"
    ] = filename


    stored_summary = get_summary(
        filename
    )


    return render_template(

        "index.html",

        message=(
            f"{filename} selected."
        ),

        pdf_count=get_pdf_count(),

        uploaded_pdfs=get_uploaded_pdfs(),

        active_pdf=filename,

        search_mode="current",

        chat_history=get_chat_history(
            filename
        ),

        summary=(
            stored_summary["summary"]
            if stored_summary
            else None
        ),

        summary_updated_at=(
            stored_summary["updated_at"]
            if stored_summary
            else None
        )

    )


# ==================================================
# VIEW PDF
# ==================================================

@app.route(
    "/pdf/<path:filename>"
)
def view_pdf(filename):

    filename = os.path.basename(
        filename
    )


    pdf_path = os.path.join(

        DATA_FOLDER,

        filename

    )


    if not os.path.exists(
        pdf_path
    ):

        return (
            "PDF not found.",
            404
        )


    return send_from_directory(

        DATA_FOLDER,

        filename

    )


# ==================================================
# GENERATE SUMMARY
# ==================================================

@app.route(
    "/summary",
    methods=["POST"]
)
def summary():

    active_pdf = session.get(
        "active_pdf"
    )


    if not active_pdf:

        return render_template(

            "index.html",

            message=(
                "Please upload or select "
                "a PDF first."
            ),

            pdf_count=get_pdf_count(),

            uploaded_pdfs=get_uploaded_pdfs(),

            active_pdf=None,

            search_mode="current",

            chat_history=[],

            summary=None

        )


    try:

        # ------------------------------------------
        # GENERATE
        # ------------------------------------------

        summary_text = generate_pdf_summary(

            active_pdf

        )


        # ------------------------------------------
        # SAVE
        # ------------------------------------------

        save_summary(

            active_pdf,

            summary_text

        )


        message = (

            f"Summary generated for "
            f"{active_pdf}."

        )


    except Exception as e:

        print(
            "\nSUMMARY ERROR:"
        )

        print(
            type(e).__name__
        )

        print(
            str(e)
        )


        message = (

            f"Summary generation failed: "
            f"{str(e)}"

        )


        summary_text = None


    stored_summary = get_summary(
        active_pdf
    )


    return render_template(

        "index.html",

        message=message,

        question="",

        answer=None,

        source_pages=[],

        pdf_count=get_pdf_count(),

        uploaded_pdfs=get_uploaded_pdfs(),

        active_pdf=active_pdf,

        search_mode="current",

        chat_history=get_chat_history(
            active_pdf
        ),

        summary=(
            stored_summary["summary"]
            if stored_summary
            else summary_text
        ),

        summary_updated_at=(
            stored_summary["updated_at"]
            if stored_summary
            else None
        )

    )


# ==================================================
# CLEAR CHAT
# ==================================================

@app.route(
    "/clear-chat",
    methods=["POST"]
)
def clear_chat():

    active_pdf = session.get(
        "active_pdf"
    )


    search_mode = request.form.get(
        "search_mode",
        "current"
    )


    if search_mode == "all":

        history_key = (
            ALL_PDF_HISTORY_KEY
        )

    else:

        history_key = active_pdf


    if history_key:

        delete_pdf_history(
            history_key
        )


    return render_template(

        "index.html",

        message="Chat history cleared.",

        pdf_count=get_pdf_count(),

        uploaded_pdfs=get_uploaded_pdfs(),

        active_pdf=active_pdf,

        search_mode=search_mode,

        chat_history=[],

        summary=(
            get_summary(active_pdf)["summary"]
            if (
                active_pdf
                and
                get_summary(active_pdf)
            )
            else None
        )

    )


# ==================================================
# DELETE PDF
# ==================================================

@app.route(
    "/delete/<path:filename>",
    methods=["POST"]
)
def delete_pdf(filename):

    filename = os.path.basename(
        filename
    )


    pdf_path = os.path.join(

        DATA_FOLDER,

        filename

    )


    try:

        # ------------------------------------------
        # DELETE VECTOR CHUNKS
        # ------------------------------------------

        existing = collection.get(

            where={
                "source":
                    filename
            }

        )


        ids = existing.get(
            "ids",
            []
        )


        if ids:

            collection.delete(
                ids=ids
            )

        rebuild_bm25()


        # ------------------------------------------
        # DELETE FILE
        # ------------------------------------------

        if os.path.exists(
            pdf_path
        ):

            os.remove(
                pdf_path
            )


        # ------------------------------------------
        # DELETE CHAT
        # ------------------------------------------

        delete_pdf_history(
            filename
        )


        # ------------------------------------------
        # DELETE SUMMARY
        # ------------------------------------------

        delete_pdf_summary(
            filename
        )


        # ------------------------------------------
        # CLEAR ACTIVE PDF
        # ------------------------------------------

        if session.get(
            "active_pdf"
        ) == filename:

            session.pop(
                "active_pdf",
                None
            )


        message = (
            f"{filename} deleted successfully."
        )


    except Exception as e:

        print(
            "\nDELETE ERROR:"
        )

        print(
            type(e).__name__
        )

        print(
            str(e)
        )


        message = (
            f"Error deleting PDF: "
            f"{str(e)}"
        )


    active_pdf = session.get(
        "active_pdf"
    )


    return render_template(

        "index.html",

        message=message,

        pdf_count=get_pdf_count(),

        uploaded_pdfs=get_uploaded_pdfs(),

        active_pdf=active_pdf,

        search_mode="current",

        chat_history=(
            get_chat_history(
                active_pdf
            )
            if active_pdf
            else []
        ),

        summary=(
            get_summary(active_pdf)["summary"]
            if (
                active_pdf
                and
                get_summary(active_pdf)
            )
            else None
        )

    )


# ==================================================
# CLEAR ALL
# ==================================================

@app.route(
    "/clear-all",
    methods=["POST"]
)
def clear_all():

    try:

        # ------------------------------------------
        # DELETE VECTOR DATA
        # ------------------------------------------

        existing = collection.get()


        ids = existing.get(
            "ids",
            []
        )


        if ids:

            collection.delete(
                ids=ids
            )

        rebuild_bm25()


        # ------------------------------------------
        # DELETE PDF FILES
        # ------------------------------------------

        deleted = 0


        for filename in os.listdir(
            DATA_FOLDER
        ):


            if filename.lower().endswith(
                ".pdf"
            ):


                pdf_path = os.path.join(

                    DATA_FOLDER,

                    filename

                )


                if os.path.isfile(
                    pdf_path
                ):


                    os.remove(
                        pdf_path
                    )


                    deleted += 1


        # ------------------------------------------
        # DELETE CHAT
        # ------------------------------------------

        delete_all_history()


        # ------------------------------------------
        # DELETE SUMMARIES
        # ------------------------------------------

        delete_all_summaries()


        # ------------------------------------------
        # CLEAR SESSION
        # ------------------------------------------

        session.pop(
            "active_pdf",
            None
        )


        message = (

            f"All PDFs cleared successfully. "
            f"{deleted} PDF(s) removed."

        )


    except Exception as e:

        print(
            "\nCLEAR ALL ERROR:"
        )

        print(
            type(e).__name__
        )

        print(
            str(e)
        )


        message = (

            f"Error clearing PDFs: "
            f"{str(e)}"

        )


    return render_template(

        "index.html",

        message=message,

        pdf_count=get_pdf_count(),

        uploaded_pdfs=get_uploaded_pdfs(),

        active_pdf=None,

        search_mode="current",

        chat_history=[],

        summary=None

    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    app.run(

        debug=False,

        host="127.0.0.1",

        port=5000

    )