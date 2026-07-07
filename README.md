# 📚 RAG Chat with Documents

A complete Retrieval-Augmented Generation (RAG) system built from scratch — no frameworks, no black boxes.

## What This Project Does

Drop PDFs or text files in the `documents/` folder. Ask questions in a browser. Get answers with citations from your actual documents — not from the LLM's training data.

## Architecture (How RAG Works)

```
INGEST (one-time):
  documents/ → loader.py → chunker.py → embedder.py → chroma_db/
     PDFs       Load        Split into    Convert to    Vector DB
                 text       ~500 char     vectors       (persistent)
                            chunks        (Ollama)

ASK (per question):
  Question → embedder.py → retriever.py → generator.py → Answer
             "What is     Convert to    Find top 5     LLM reads
              revenue?"    vector        closest         chunks +
                           chunks        + answer with
                                            citations
```

## Modules Explained

| Module | File | What It Does | Why It Exists |
|--------|------|-------------|--------------|
| **Loader** | `loader.py` | Reads PDFs/text from folder | Can't search what you haven't loaded |
| **Chunker** | `chunker.py` | Splits documents into ~500 char pieces | Big documents blur search; small chunks are precise |
| **Embedder** | `embedder.py` | Converts text → vectors using Ollama | Computers search numbers, not text |
| **Vector Store** | `embedder.py` | Stores vectors in Chroma (local) | Need persistent storage + similarity search |
| **Retriever** | `retriever.py` | Finds relevant chunks for a question | The "R" in RAG — Retrieval |
| **Generator** | `generator.py` | LLM answers from retrieved context | The "G" in RAG — Generation |
| **Main CLI** | `main.py` | Ties everything together (terminal) | Command-line interface |
| **Web App** | `web_app.py` | Browser interface with FastAPI | Visual UI for non-technical users |

## How to Run

### 1. Install Dependencies
```bash
cd ~/Desktop/rag-chat-docs
source venv/bin/activate
pip install chromadb pymupdf requests fastapi uvicorn
```

### 2. Add Documents
Put `.pdf`, `.txt`, or `.md` files in the `documents/` folder.

### 3. Ingest (embed documents into vector DB)
```bash
cd app
python main.py ingest
```

### 4. Ask Questions (CLI)
```bash
python main.py ask "What is the revenue model?"
python main.py interactive   # Interactive mode
```

### 5. Web App (browser interface)
```bash
python web_app.py
# Open: http://localhost:8000
# API docs: http://localhost:8000/docs
```

## Key Concepts for Students

### Why chunking? 
A 50-page PDF as one vector = a "blur" — close to everything, specific to nothing. Breaking into ~500 char chunks makes search precise.

### Why overlap?
If you cut at exactly 500 chars, you might split a sentence: "...revenue is based on" | "subscription fees...". Overlap (50 chars) ensures the boundary info survives in both chunks.

### Why embeddings?
Computers can't "search" text. Embeddings convert text to numbers where similar MEANING = similar numbers. "Revenue" and "money" have similar vectors even though they're different words.

### Why "answer only from context"?
Without this instruction, the LLM mixes its training data with your documents and hallucinates. The system prompt forces it to answer from retrieved chunks only — and cite sources.

### Why same embedding model for questions and docs?
If documents are embedded with model A and questions with model B, the vectors live in different "spaces" — similarity search is meaningless. Same model = same space = meaningful search.

## Tech Stack
- **Embedding model:** Ollama `nomic-embed-text` (local, free, 768-dim vectors)
- **Vector database:** Chroma (local file-based, zero infrastructure)
- **LLM:** GLM-5.2 Cloud via proxy (OpenAI-compatible API)
- **Web framework:** FastAPI
- **PDF parsing:** PyMuPDF (fitz)

## File Structure
```
rag-chat-docs/
├── documents/           # Drop your PDFs/text here
│   ├── acme_business_report.txt
│   └── acme_product_roadmap.txt
├── chroma_db/           # Vector database (auto-created)
├── venv/                # Python virtual environment
└── app/
    ├── loader.py        # Module 1: Load documents
    ├── chunker.py       # Module 2: Split into chunks
    ├── embedder.py      # Module 3: Embed + store vectors
    ├── retriever.py     # Module 4: Search relevant chunks
    ├── generator.py     # Module 5: LLM answer generation
    ├── main.py          # Module 6: CLI interface
    └── web_app.py       # Module 7: Web interface
```