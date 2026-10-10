import json
import os
import re
import sys
from datetime import datetime

# --------------------------------------------------
# Make project root importable when running:
# python evaluation/evaluate.py
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --------------------------------------------------
# Import the existing RAG application
# --------------------------------------------------

import app


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

QUESTIONS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "questions.json"
)

RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evaluation_results.json"
)

REPORT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evaluation_report.json"
)

GROQ_MODEL = "openai/gpt-oss-20b"


# --------------------------------------------------
# LOAD QUESTIONS
# --------------------------------------------------

def load_questions():
    """Load benchmark questions from questions.json."""

    if not os.path.exists(QUESTIONS_FILE):
        raise FileNotFoundError(
            f"Questions file not found: {QUESTIONS_FILE}"
        )

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "questions.json must contain a JSON array."
        )

    return questions


# --------------------------------------------------
# TEXT NORMALIZATION
# --------------------------------------------------

def normalize_text(text):
    """Normalize text for keyword matching."""

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# --------------------------------------------------
# ANSWER KEYWORD CHECK
# --------------------------------------------------

def calculate_keyword_coverage(
    answer,
    keywords
):
    """
    Calculate how many expected keywords are present
    in the generated answer.

    Returns:
        float between 0 and 1
    """

    if not keywords:
        return 0.0

    normalized_answer = normalize_text(
        answer
    )

    matches = 0

    for keyword in keywords:

        normalized_keyword = normalize_text(
            keyword
        )

        if not normalized_keyword:
            continue

        if normalized_keyword in normalized_answer:
            matches += 1

    return matches / len(keywords)


# --------------------------------------------------
# EXPECTED ANSWER CHECK
# --------------------------------------------------

def expected_answer_match(
    answer,
    expected_answer
):
    """
    Check whether the expected answer appears
    in the generated answer.

    This is intentionally conservative.
    """

    if not expected_answer:
        return False

    normalized_answer = normalize_text(
        answer
    )

    normalized_expected = normalize_text(
        expected_answer
    )

    if not normalized_answer:
        return False

    if not normalized_expected:
        return False

    return normalized_expected in normalized_answer


# --------------------------------------------------
# REFUSAL DETECTION
# --------------------------------------------------

def is_refusal(answer):
    """
    Detect whether the RAG assistant refused to answer
    because the information was not found.
    """

    if not answer:
        return False

    normalized = normalize_text(
        answer
    )

    refusal_patterns = [

        "i could not find the answer "
        "in the uploaded pdfs",

        "i couldn't find the answer "
        "in the uploaded pdfs",

        "i could not find the answer "
        "in the uploaded pdf",

        "i couldn't find the answer "
        "in the uploaded pdf",

        "not found in the uploaded pdfs",

        "not available in the uploaded pdfs",

        "cannot be found in the uploaded pdfs",

        "not provided in the uploaded pdfs",

        "the uploaded pdfs do not contain",

        "the document does not contain",

        "not mentioned in the uploaded pdfs"

    ]

    for pattern in refusal_patterns:

        if pattern in normalized:
            return True

    return False


# --------------------------------------------------
# SOURCE EXTRACTION
# --------------------------------------------------

def extract_sources(metadatas):
    """
    Extract unique PDF source names from retrieved chunks.
    """

    sources = []

    seen = set()

    for metadata in metadatas:

        if not metadata:
            continue

        source = metadata.get(
            "source"
        )

        if not source:
            continue

        source = str(source)

        if source not in seen:

            seen.add(source)

            sources.append(source)

    return sources


# --------------------------------------------------
# PAGE EXTRACTION
# --------------------------------------------------

def extract_source_pages(metadatas):
    """
    Return unique source/page combinations.
    """

    pages = []

    seen = set()

    for metadata in metadatas:

        if not metadata:
            continue

        source = metadata.get(
            "source"
        )

        page = metadata.get(
            "page"
        )

        if source is None:
            continue

        key = (
            str(source),
            str(page)
        )

        if key in seen:
            continue

        seen.add(key)

        pages.append({
            "source": str(source),
            "page": page
        })

    return pages


# --------------------------------------------------
# RETRIEVAL ACCURACY
# --------------------------------------------------

def retrieval_correct(
    retrieved_sources,
    expected_sources
):
    """
    Retrieval is considered correct when at least one
    expected source PDF appears in the reranked results.
    """

    if not expected_sources:
        return None

    retrieved_normalized = {
        normalize_text(source)
        for source in retrieved_sources
    }

    expected_normalized = {
        normalize_text(source)
        for source in expected_sources
    }

    return bool(
        retrieved_normalized
        & expected_normalized
    )


# --------------------------------------------------
# CITATION / SOURCE ACCURACY
# --------------------------------------------------

def citation_correct(
    retrieved_sources,
    expected_sources
):
    """
    Check whether the retrieved source metadata includes
    at least one expected PDF.

    Since the current application displays sources outside
    the generated answer, this evaluates the RAG source
    metadata rather than requiring the LLM to type citations.
    """

    if not expected_sources:
        return None

    return retrieval_correct(
        retrieved_sources,
        expected_sources
    )


# --------------------------------------------------
# ANSWER ACCURACY
# --------------------------------------------------

def answer_correct(
    answer,
    question_data
):
    """
    Evaluate answerable and unanswerable questions.

    For answerable questions:
        - expected answer exact substring match OR
        - keyword coverage >= 0.5

    For unanswerable questions:
        - refusal should be detected
    """

    answerable = question_data.get(
        "answerable",
        True
    )

    expected_answer = question_data.get(
        "expected_answer"
    )

    keywords = question_data.get(
        "keywords",
        []
    )

    if not answerable:

        return is_refusal(
            answer
        )

    # Exact expected answer match
    if expected_answer:

        if expected_answer_match(
            answer,
            expected_answer
        ):
            return True

    # Keyword-based evaluation
    coverage = calculate_keyword_coverage(
        answer,
        keywords
    )

    if coverage >= 0.5:
        return True

    return False


# --------------------------------------------------
# QUESTION EVALUATION
# --------------------------------------------------

def evaluate_question(
    question_data,
    index,
    total
):
    """
    Run one benchmark question through the actual
    hybrid RAG pipeline.
    """

    question_id = question_data.get(
        "id",
        index
    )

    question = question_data.get(
        "question",
        ""
    ).strip()

    expected_sources = question_data.get(
        "expected_sources",
        []
    )

    question_type = question_data.get(
        "type",
        "unknown"
    )

    if not question:

        return {
            "id": question_id,
            "question": question,
            "type": question_type,
            "error": "Empty question",
            "success": False
        }

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"[{index}/{total}] Question #{question_id}"
    )

    print(
        f"Type: {question_type}"
    )

    print(
        f"Question: {question}"
    )

    try:

        # ------------------------------------------
        # STEP 1
        # QUESTION EMBEDDING
        # ------------------------------------------

        question_embedding = (
            app.embedding_model
            .encode(question)
            .tolist()
        )

        # ------------------------------------------
        # STEP 2
        # HYBRID SEARCH
        # ------------------------------------------

        print(
            "Running hybrid retrieval..."
        )

        hybrid_candidates = app.hybrid_search(
            question,
            question_embedding,
            source=None
        )

        if not hybrid_candidates:

            return {
                "id": question_id,
                "question": question,
                "type": question_type,
                "expected_sources": expected_sources,
                "retrieved_sources": [],
                "retrieved_pages": [],
                "answer": (
                    "I could not find the answer "
                    "in the uploaded PDFs."
                ),
                "refused": True,
                "retrieval_correct": (
                    False
                    if expected_sources
                    else None
                ),
                "answer_correct": (
                    not question_data.get(
                        "answerable",
                        True
                    )
                ),
                "citation_correct": (
                    False
                    if expected_sources
                    else None
                ),
                "success": True
            }

        # ------------------------------------------
        # STEP 3
        # PREPARE CANDIDATES
        # ------------------------------------------

        documents = [
            item["document"]
            for item in hybrid_candidates
        ]

        metadatas = [
            item["metadata"]
            for item in hybrid_candidates
        ]

        # ------------------------------------------
        # STEP 4
        # CROSS ENCODER RERANKING
        # ------------------------------------------

        print(
            "Running cross-encoder reranking..."
        )

        (
            final_documents,
            final_metadatas
        ) = app.rerank_documents(
            question,
            documents,
            metadatas
        )

        # ------------------------------------------
        # STEP 5
        # SOURCES
        # ------------------------------------------

        retrieved_sources = extract_sources(
            final_metadatas
        )

        retrieved_pages = extract_source_pages(
            final_metadatas
        )

        print(
            "Retrieved sources:",
            retrieved_sources
        )

        # ------------------------------------------
        # STEP 6
        # BUILD CONTEXT
        # ------------------------------------------

        context_parts = []

        for (
            document,
            metadata
        ) in zip(
            final_documents,
            final_metadatas
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

        context = "\n".join(
            context_parts
        )

        # ------------------------------------------
        # STEP 7
        # GENERATE ANSWER
        # ------------------------------------------

        print(
            "Generating answer with Groq..."
        )

        prompt = f"""
You are a Retrieval-Augmented Generation
(RAG) document question-answering assistant.

Answer the user's question using ONLY the
retrieved PDF context below.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. Use only the supplied PDF context.
4. Answer directly and clearly.
5. Mention the source PDF when useful.
6. If the answer cannot be found in the context,
   say exactly:

"I could not find the answer in the uploaded PDFs."

PDF CONTEXT:

{context}

USER QUESTION:

{question}
"""

        response = (
            app.groq_client
            .chat
            .completions
            .create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer only from the "
                            "supplied PDF context."
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

        if answer is None:
            answer = ""

        answer = answer.strip()

        # ------------------------------------------
        # STEP 8
        # METRICS
        # ------------------------------------------

        retrieval_result = retrieval_correct(
            retrieved_sources,
            expected_sources
        )

        answer_result = answer_correct(
            answer,
            question_data
        )

        citation_result = citation_correct(
            retrieved_sources,
            expected_sources
        )

        refused = is_refusal(
            answer
        )

        keyword_coverage = (
            calculate_keyword_coverage(
                answer,
                question_data.get(
                    "keywords",
                    []
                )
            )
            if question_data.get(
                "answerable",
                True
            )
            else None
        )

        print(
            "Answer:",
            answer
        )

        print(
            "Retrieval correct:",
            retrieval_result
        )

        print(
            "Answer correct:",
            answer_result
        )

        print(
            "Citation correct:",
            citation_result
        )

        print(
            "Refused:",
            refused
        )

        result = {

            "id": question_id,

            "question": question,

            "type": question_type,

            "answerable":
                question_data.get(
                    "answerable",
                    True
                ),

            "expected_answer":
                question_data.get(
                    "expected_answer"
                ),

            "expected_keywords":
                question_data.get(
                    "keywords",
                    []
                ),

            "expected_sources":
                expected_sources,

            "retrieved_sources":
                retrieved_sources,

            "retrieved_pages":
                retrieved_pages,

            "answer":
                answer,

            "refused":
                refused,

            "keyword_coverage":
                keyword_coverage,

            "retrieval_correct":
                retrieval_result,

            "answer_correct":
                answer_result,

            "citation_correct":
                citation_result,

            "success":
                True
        }

        return result

    except Exception as e:

        print(
            "\nERROR:"
        )

        print(
            type(e).__name__
        )

        print(
            str(e)
        )

        return {

            "id":
                question_id,

            "question":
                question,

            "type":
                question_type,

            "expected_sources":
                expected_sources,

            "error_type":
                type(e).__name__,

            "error":
                str(e),

            "success":
                False
        }


# --------------------------------------------------
# METRIC HELPERS
# --------------------------------------------------

def calculate_rate(
    values
):
    """
    Calculate percentage for boolean values.

    None values are ignored.
    """

    usable = [
        value
        for value in values
        if value is not None
    ]

    if not usable:
        return 0.0

    correct = sum(
        1
        for value in usable
        if value is True
    )

    return (
        correct / len(usable)
    ) * 100


# --------------------------------------------------
# BUILD REPORT
# --------------------------------------------------

def build_report(
    results
):

    successful_results = [
        result
        for result in results
        if result.get(
            "success"
        )
    ]

    failed_results = [
        result
        for result in results
        if not result.get(
            "success"
        )
    ]

    # ------------------------------------------
    # ALL QUESTION METRICS
    # ------------------------------------------

    retrieval_accuracy = calculate_rate(
        [
            result.get(
                "retrieval_correct"
            )
            for result in successful_results
        ]
    )

    answer_accuracy = calculate_rate(
        [
            result.get(
                "answer_correct"
            )
            for result in successful_results
        ]
    )

    citation_accuracy = calculate_rate(
        [
            result.get(
                "citation_correct"
            )
            for result in successful_results
        ]
    )

    # ------------------------------------------
    # REFUSAL ACCURACY
    # ------------------------------------------

    unanswerable_results = [

        result

        for result in successful_results

        if result.get(
            "answerable"
        ) is False

    ]

    refusal_accuracy = calculate_rate(
        [
            result.get(
                "refused"
            )
            for result in unanswerable_results
        ]
    )

    # ------------------------------------------
    # ANSWERABLE QUESTIONS
    # ------------------------------------------

    answerable_results = [

        result

        for result in successful_results

        if result.get(
            "answerable"
        ) is True

    ]

    answerable_accuracy = calculate_rate(
        [
            result.get(
                "answer_correct"
            )
            for result in answerable_results
        ]
    )

    # ------------------------------------------
    # QUESTION TYPE METRICS
    # ------------------------------------------

    by_type = {}

    question_types = sorted(
        {
            result.get(
                "type",
                "unknown"
            )
            for result in results
        }
    )

    for question_type in question_types:

        type_results = [

            result

            for result in successful_results

            if result.get(
                "type"
            ) == question_type

        ]

        by_type[
            question_type
        ] = {

            "total":
                len(type_results),

            "retrieval_accuracy":
                calculate_rate(
                    [
                        result.get(
                            "retrieval_correct"
                        )
                        for result in type_results
                    ]
                ),

            "answer_accuracy":
                calculate_rate(
                    [
                        result.get(
                            "answer_correct"
                        )
                        for result in type_results
                    ]
                ),

            "citation_accuracy":
                calculate_rate(
                    [
                        result.get(
                            "citation_correct"
                        )
                        for result in type_results
                    ]
                )
        }

    # ------------------------------------------
    # OVERALL SCORE
    # ------------------------------------------

    metric_values = [
        retrieval_accuracy,
        answer_accuracy,
        citation_accuracy
    ]

    # Include refusal accuracy only when there
    # are unanswerable questions.
    if unanswerable_results:

        metric_values.append(
            refusal_accuracy
        )

    overall_score = (
        sum(metric_values)
        / len(metric_values)
        if metric_values
        else 0.0
    )

    # ------------------------------------------
    # FINAL REPORT
    # ------------------------------------------

    report = {

        "evaluation_timestamp":
            datetime.now().isoformat(),

        "total_questions":
            len(results),

        "successful_questions":
            len(successful_results),

        "failed_questions":
            len(failed_results),

        "answerable_questions":
            len(answerable_results),

        "unanswerable_questions":
            len(unanswerable_results),

        "retrieval_accuracy_percent":
            round(
                retrieval_accuracy,
                2
            ),

        "answer_accuracy_percent":
            round(
                answer_accuracy,
                2
            ),

        "answerable_accuracy_percent":
            round(
                answerable_accuracy,
                2
            ),

        "refusal_accuracy_percent":
            round(
                refusal_accuracy,
                2
            ),

        "citation_accuracy_percent":
            round(
                citation_accuracy,
                2
            ),

        "overall_score_percent":
            round(
                overall_score,
                2
            ),

        "metrics_definition": {

            "retrieval_accuracy":
                (
                    "Percentage of questions where at least "
                    "one expected source PDF appeared in the "
                    "final reranked chunks."
                ),

            "answer_accuracy":
                (
                    "Percentage of questions where the "
                    "generated answer matched the expected "
                    "answer or achieved at least 50% keyword "
                    "coverage, with unanswerable questions "
                    "counted according to refusal behavior."
                ),

            "answerable_accuracy":
                (
                    "Answer accuracy calculated only for "
                    "questions marked answerable=true."
                ),

            "refusal_accuracy":
                (
                    "Percentage of unanswerable questions for "
                    "which the assistant correctly refused."
                ),

            "citation_accuracy":
                (
                    "Percentage of answerable/source-backed "
                    "questions where the expected PDF source "
                    "appeared in the final retrieved metadata."
                ),

            "overall_score":
                (
                    "Mean of retrieval accuracy, answer accuracy, "
                    "citation accuracy, and refusal accuracy when "
                    "unanswerable questions are present."
                )
        },

        "results_by_question_type":
            by_type,

        "failed_question_ids":
            [
                result.get("id")
                for result in failed_results
            ]
    }

    return report


# --------------------------------------------------
# SAVE JSON
# --------------------------------------------------

def save_json(
    filename,
    data
):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


# --------------------------------------------------
# PRINT REPORT
# --------------------------------------------------

def print_report(
    report
):

    print("\n")
    print("=" * 70)
    print("RAG EVALUATION REPORT")
    print("=" * 70)

    print(
        f"Total questions: "
        f"{report['total_questions']}"
    )

    print(
        f"Successful: "
        f"{report['successful_questions']}"
    )

    print(
        f"Failed: "
        f"{report['failed_questions']}"
    )

    print(
        f"Answerable: "
        f"{report['answerable_questions']}"
    )

    print(
        f"Unanswerable: "
        f"{report['unanswerable_questions']}"
    )

    print()
    print(
        f"Retrieval Accuracy: "
        f"{report['retrieval_accuracy_percent']:.2f}%"
    )

    print(
        f"Answer Accuracy: "
        f"{report['answer_accuracy_percent']:.2f}%"
    )

    print(
        f"Answerable Accuracy: "
        f"{report['answerable_accuracy_percent']:.2f}%"
    )

    print(
        f"Refusal Accuracy: "
        f"{report['refusal_accuracy_percent']:.2f}%"
    )

    print(
        f"Citation Accuracy: "
        f"{report['citation_accuracy_percent']:.2f}%"
    )

    print(
        f"Overall Score: "
        f"{report['overall_score_percent']:.2f}%"
    )

    print()
    print("=" * 70)

    print(
        "RESULTS BY QUESTION TYPE"
    )

    print("=" * 70)

    for (
        question_type,
        metrics
    ) in report[
        "results_by_question_type"
    ].items():

        print()
        print(
            question_type.upper()
        )

        print(
            f"  Total: "
            f"{metrics['total']}"
        )

        print(
            f"  Retrieval: "
            f"{metrics['retrieval_accuracy']:.2f}%"
        )

        print(
            f"  Answer: "
            f"{metrics['answer_accuracy']:.2f}%"
        )

        print(
            f"  Citation: "
            f"{metrics['citation_accuracy']:.2f}%"
        )

    print()
    print("=" * 70)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print(
        "\nStarting RAG evaluation..."
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Questions file: {QUESTIONS_FILE}"
    )

    # ------------------------------------------
    # LOAD QUESTIONS
    # ------------------------------------------

    questions = load_questions()

    print(
        f"Loaded {len(questions)} benchmark questions."
    )

    # ------------------------------------------
    # SAFETY CHECK
    # ------------------------------------------

    if app.collection.count() == 0:

        raise RuntimeError(
            "ChromaDB contains no documents. "
            "Upload and index at least one PDF before "
            "running the evaluation."
        )

    print(
        f"ChromaDB chunks available: "
        f"{app.collection.count()}"
    )

    # ------------------------------------------
    # RUN EVALUATION
    # ------------------------------------------

    results = []

    total = len(
        questions
    )

    for index, question_data in enumerate(
        questions,
        start=1
    ):

        result = evaluate_question(
            question_data,
            index,
            total
        )

        results.append(
            result
        )

    # ------------------------------------------
    # BUILD REPORT
    # ------------------------------------------

    report = build_report(
        results
    )

    # ------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------

    save_json(
        RESULTS_FILE,
        results
    )

    save_json(
        REPORT_FILE,
        report
    )

    # ------------------------------------------
    # PRINT REPORT
    # ------------------------------------------

    print_report(
        report
    )

    print()
    print(
        f"Detailed results saved to:"
    )

    print(
        RESULTS_FILE
    )

    print()
    print(
        f"Evaluation report saved to:"
    )

    print(
        REPORT_FILE
    )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    main()