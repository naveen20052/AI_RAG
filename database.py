import sqlite3
import json
from datetime import datetime


DATABASE_NAME = "chat_history.db"


# ==================================================
# DATABASE CONNECTION
# ==================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_NAME
    )

    connection.row_factory = sqlite3.Row

    return connection


# ==================================================
# INITIALIZE DATABASE
# ==================================================

def init_database():

    connection = get_connection()

    cursor = connection.cursor()

    # ----------------------------------------------
    # CHAT HISTORY TABLE
    # ----------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            pdf_name TEXT NOT NULL,

            question TEXT NOT NULL,

            answer TEXT NOT NULL,

            search_mode TEXT NOT NULL,

            sources TEXT,

            created_at TEXT NOT NULL

        )
    """)


    # ----------------------------------------------
    # SUMMARY TABLE
    # ----------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_summaries (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            pdf_name TEXT UNIQUE NOT NULL,

            summary TEXT NOT NULL,

            updated_at TEXT NOT NULL

        )
    """)


    connection.commit()

    connection.close()


# ==================================================
# SAVE CHAT
# ==================================================

def save_chat(
    pdf_name,
    question,
    answer,
    search_mode,
    sources
):

    connection = get_connection()

    cursor = connection.cursor()


    source_json = json.dumps(
        sources,
        ensure_ascii=False
    )


    created_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    cursor.execute(
        """
        INSERT INTO chat_history
        (
            pdf_name,
            question,
            answer,
            search_mode,
            sources,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            pdf_name,
            question,
            answer,
            search_mode,
            source_json,
            created_at
        )
    )


    connection.commit()

    connection.close()


# ==================================================
# GET CHAT HISTORY
# ==================================================

def get_chat_history(pdf_name):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT
            id,
            pdf_name,
            question,
            answer,
            search_mode,
            sources,
            created_at
        FROM chat_history
        WHERE pdf_name = ?
        ORDER BY id ASC
        """,
        (pdf_name,)
    )


    rows = cursor.fetchall()

    connection.close()


    history = []


    for row in rows:

        sources = []


        if row["sources"]:

            try:

                sources = json.loads(
                    row["sources"]
                )

            except json.JSONDecodeError:

                sources = []


        history.append({

            "id": row["id"],

            "pdf": row["pdf_name"],

            "question": row["question"],

            "answer": row["answer"],

            "mode": row["search_mode"],

            "sources": sources,

            "created_at": row["created_at"]

        })


    return history


# ==================================================
# DELETE PDF CHAT HISTORY
# ==================================================

def delete_pdf_history(pdf_name):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        DELETE FROM chat_history
        WHERE pdf_name = ?
        """,
        (pdf_name,)
    )


    connection.commit()

    connection.close()


# ==================================================
# DELETE ALL CHAT HISTORY
# ==================================================

def delete_all_history():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        "DELETE FROM chat_history"
    )


    connection.commit()

    connection.close()


# ==================================================
# SAVE SUMMARY
# ==================================================

def save_summary(
    pdf_name,
    summary
):

    connection = get_connection()

    cursor = connection.cursor()


    updated_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    cursor.execute(
        """
        INSERT INTO document_summaries
        (
            pdf_name,
            summary,
            updated_at
        )
        VALUES (?, ?, ?)

        ON CONFLICT(pdf_name)
        DO UPDATE SET
            summary = excluded.summary,
            updated_at = excluded.updated_at
        """,
        (
            pdf_name,
            summary,
            updated_at
        )
    )


    connection.commit()

    connection.close()


# ==================================================
# GET SUMMARY
# ==================================================

def get_summary(pdf_name):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT
            summary,
            updated_at
        FROM document_summaries
        WHERE pdf_name = ?
        """,
        (pdf_name,)
    )


    row = cursor.fetchone()

    connection.close()


    if not row:

        return None


    return {

        "summary":
            row["summary"],

        "updated_at":
            row["updated_at"]

    }


# ==================================================
# DELETE PDF SUMMARY
# ==================================================

def delete_pdf_summary(pdf_name):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        DELETE FROM document_summaries
        WHERE pdf_name = ?
        """,
        (pdf_name,)
    )


    connection.commit()

    connection.close()


# ==================================================
# DELETE ALL SUMMARIES
# ==================================================

def delete_all_summaries():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        "DELETE FROM document_summaries"
    )


    connection.commit()

    connection.close()