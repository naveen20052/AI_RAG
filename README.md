# AI_RAG – PDF Question Answering Assistant

A Python-based Retrieval-Augmented Generation (RAG) application that helps users upload PDF documents and ask questions about their content.

## Features

- Upload and manage PDF documents
- Extract and split document text into chunks
- Generate embeddings for document retrieval
- Ask questions about uploaded documents
- View conversation history
- Evaluate question-answering performance using a test dataset

## Technology Stack

- Python
- Flask
- HTML and CSS
- Groq API
- ChromaDB
- Git and GitHub

## Project Structure

- `app.py` – Web application entry point
- `read_pdf.py` – PDF text extraction
- `chunk_text.py` – Text chunking
- `create_embeddings.py` – Embedding generation
- `store_embeddings.py` – Embedding storage
- `retrieve.py` – Relevant context retrieval
- `chat.py` – Question-answering logic
- `database.py` – Database operations
- `evaluation/` – Evaluation files and results
- `templates/` – HTML templates
- `static/` – CSS styles
- `screenshots/` – Application screenshot

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required packages using `pip install -r requirements.txt`.
4. Create a `.env` file and configure the API key required by the application.
5. Start the application using `python app.py`.
6. Open `http://127.0.0.1:5000` in your browser.

## Security

Keep API keys in environment variables. Never commit `.env`, private documents, or sensitive conversation data.

## Demo

See `screenshots/rag-demo.png` for an application screenshot.