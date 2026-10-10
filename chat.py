import os

import chromadb

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


load_dotenv()


CHROMA_FOLDER = "chroma_db"

COLLECTION_NAME = "pdf_documents"


api_key = os.getenv(
    "GROQ_API_KEY"
)


if not api_key:

    print(
        "❌ GROQ_API_KEY not found."
    )

    exit()


groq_client = Groq(

    api_key=api_key,

    timeout=60.0,

    max_retries=1
)


print(
    "Loading embedding model..."
)


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print(
    "✅ Embedding model loaded."
)


client = chromadb.PersistentClient(

    path=CHROMA_FOLDER
)


collection = client.get_or_create_collection(

    name=COLLECTION_NAME
)


print(
    f"📚 Stored chunks: "
    f"{collection.count()}"
)


while True:

    question = input(
        "\nAsk a question "
        "(type 'exit' to stop): "
    ).strip()


    if question.lower() == "exit":

        print(
            "Goodbye!"
        )

        break


    if not question:

        continue


    try:

        print(
            "\n[1/4] Creating embedding..."
        )


        question_embedding = (

            model
            .encode(question)
            .tolist()

        )


        print(
            "[2/4] Searching all PDFs..."
        )


        results = collection.query(

            query_embeddings=[
                question_embedding
            ],

            n_results=5,

            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )


        documents = results[
            "documents"
        ][0]


        metadatas = results[
            "metadatas"
        ][0]


        distances = results[
            "distances"
        ][0]


        context_parts = []

        sources = []


        for (
            document,
            metadata,
            distance
        ) in zip(
            documents,
            metadatas,
            distances
        ):


            if distance > 1.5:

                continue


            source = metadata.get(
                "source",
                "Unknown PDF"
            )


            page = metadata.get(
                "page",
                "Unknown"
            )


            context_parts.append(

                f"""
PDF: {source}

Page: {page}

Content:
{document}
"""

            )


            source_info = (
                f"{source} - Page {page}"
            )


            if source_info not in sources:

                sources.append(
                    source_info
                )


        if not context_parts:

            print(
                "\nI could not find the answer "
                "in the uploaded PDFs."
            )

            continue


        context = "\n".join(
            context_parts
        )


        print(
            "[3/4] Asking Groq..."
        )


        prompt = f"""
You are a multi-PDF question-answering assistant.

Use ONLY the information in the PDF context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If multiple PDFs are relevant,
   combine their information.
4. Give a clear answer.
5. If the answer is not present,
   say:

"I could not find the answer in the uploaded PDFs."


PDF CONTEXT:

{context}


USER QUESTION:

{question}
"""


        response = (

            groq_client
            .chat
            .completions
            .create(

                model="openai/gpt-oss-20b",

                messages=[

                    {
                        "role": "system",
                        "content": (
                            "Answer only using "
                            "the supplied PDF context."
                        )
                    },

                    {
                        "role": "user",
                        "content": prompt
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


        print(
            "\n[4/4] Answer generated."
        )


        print(
            "\n=============================="
        )

        print(
            "ANSWER"
        )

        print(
            "=============================="
        )

        print(
            answer
        )


        print(
            "\nSOURCES:"
        )


        for source in sources:

            print(
                f"- {source}"
            )


    except Exception as e:

        print(
            "\n❌ Error:"
        )

        print(
            type(e).__name__
        )

        print(
            e
        )