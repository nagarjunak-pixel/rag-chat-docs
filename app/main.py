"""
MODULE 6: MAIN CLI
==================
WHY THIS EXISTS:
  We have 5 modules (loader, chunker, embedder, retriever, generator)
  but no way to USE them together. main.py is the entry point that ties
  everything into a working pipeline.

WHAT IT DOES:
  Two commands:
  
  1. INGEST: Load docs → Chunk → Embed → Store in Chroma
     Run this once (or whenever you add new documents)
     
     python main.py ingest
  
  2. ASK: Question → Embed → Retrieve → Generate answer  
     Run this for every question
     
     python main.py ask "What is the revenue model?"
  
  3. STATS: Show how many chunks are stored
     python main.py stats

THE FULL RAG PIPELINE:
  
  INGEST (one-time setup):
    documents/ → loader.py → chunker.py → embedder.py → chroma_db/
       PDFs       Load        Split into    Convert to    Vector DB
                   text       chunks       vectors       (persistent)
  
  ASK (per question):
    Question → embedder.py → retriever.py → generator.py → Answer
                "What is     Convert to    Find closest    LLM reads
                 revenue?"    vector        chunks          chunks +
                                                            answers
                                                            with cites

WHY separate ingest and ask?
  Ingest is expensive (embedding 100 chunks = 100 API calls to Ollama).
  Ask is cheap (1 embedding + 1 LLM call).
  You ingest ONCE, then ask unlimited questions.
  Chroma persists to disk, so embeddings survive restarts.
"""

import sys
import os

# WHY add parent directory to path? So we can import from app/ modules
# without needing to install them as a package. Simple and student-friendly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loader import load_documents
from chunker import chunk_documents
from embedder import Embedder, VectorStore
from retriever import Retriever
from generator import AnswerGenerator


# WHY these paths as constants? So students can easily change them.
# Want to use a different documents folder? Change DOCS_PATH. Done.
DOCS_PATH = os.path.join(os.path.dirname(__file__), "..", "documents")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def ingest():
    """
    STEP 1: Load documents from documents/ folder.
    STEP 2: Chunk them into ~500 char pieces with overlap.
    STEP 3: Embed each chunk using Ollama nomic-embed-text.
    STEP 4: Store embeddings + text + metadata in Chroma.
    
    WHY run this once? Embedding is the expensive part.
    100 chunks = 100 calls to Ollama. Takes ~30 seconds.
    But Chroma persists to disk — next time you just ask questions.
    """
    print("\n" + "=" * 60)
    print("  📥 INGEST: Loading documents into vector store")
    print("=" * 60)
    
    # STEP 1: Load
    print("\n  Step 1/4: Loading documents...")
    documents = load_documents(DOCS_PATH)
    if not documents:
        print("\n  ❌ No documents found! Add PDFs/text to documents/ folder.")
        return
    
    # STEP 2: Chunk
    print("\n  Step 2/4: Chunking documents...")
    chunks = chunk_documents(documents, chunk_size=500, overlap=50)
    print(f"  ✅ Created {len(chunks)} chunks from {len(documents)} pages")
    
    # STEP 3: Embed
    print(f"\n  Step 3/4: Embedding {len(chunks)} chunks (this takes a moment)...")
    embedder = Embedder()
    
    # WHY embed one at a time? Ollama processes embeddings sequentially.
    # The progress bar shows students that something is happening —
    # embedding 100 chunks can take 30+ seconds.
    embeddings = []
    for i, chunk in enumerate(chunks):
        emb = embedder.embed(chunk["text"])
        embeddings.append(emb)
        # Progress indicator
        if (i + 1) % 10 == 0 or i == 0 or i == len(chunks) - 1:
            print(f"     Embedded {i + 1}/{len(chunks)} chunks...")
    
    # STEP 4: Store
    print(f"\n  Step 4/4: Storing in Chroma vector database...")
    store = VectorStore(db_path=CHROMA_PATH)
    store.clear()  # WHY clear? Fresh start. Remove old chunks from previous runs.
    store.add_chunks(chunks, embeddings)
    
    print(f"\n  ✅ INGEST COMPLETE!")
    print(f"     {len(chunks)} chunks stored in {CHROMA_PATH}")
    print(f"     Now run: python main.py ask \"your question\"")
    print("=" * 60)


def ask(question: str):
    """
    STEP 1: Embed the question using same model as documents.
    STEP 2: Search Chroma for top 5 most similar chunks.
    STEP 3: Send question + retrieved chunks to LLM.
    STEP 4: LLM answers from context with citations.
    
    WHY is this fast? Only 2 API calls:
      1. Embed the question (Ollama, ~0.5s)
      2. Generate answer (GLM-5.2 cloud, ~2-5s)
    Chroma search is local — milliseconds.
    """
    print("\n" + "=" * 60)
    print("  ❓ ASK: Searching documents and generating answer")
    print("=" * 60)
    
    # Initialize components
    embedder = Embedder()
    store = VectorStore(db_path=CHROMA_PATH)
    retriever = Retriever(embedder, store, top_k=5)
    generator = AnswerGenerator()
    
    # STEP 1-2: Retrieve relevant chunks
    print(f"\n  🔍 Searching for: \"{question}\"")
    context = retriever.retrieve_with_context(question)
    
    # Show what was retrieved
    results = retriever.retrieve(question)
    print(f"  📄 Found {len(results)} relevant chunks:")
    for i, r in enumerate(results, 1):
        preview = r["text"][:60].replace("\n", " ")
        print(f"     {i}. [{r['source']}, p.{r['page']}] score={r['score']:.2f} — {preview}...")
    
    # STEP 3-4: Generate answer
    print(f"\n  🧠 Generating answer from context...")
    answer = generator.generate(question, context)
    
    print(f"\n  💬 ANSWER:")
    print(f"  " + "-" * 56)
    # WHY indent each line? For clean terminal output.
    for line in answer.split("\n"):
        print(f"  {line}")
    print(f"  " + "-" * 56)
    print("=" * 60)


def stats():
    """Show how many chunks are in the vector store."""
    print("\n  📊 Vector Store Statistics:")
    store = VectorStore(db_path=CHROMA_PATH)
    count = store.collection.count()
    print(f"     Total chunks: {count}")
    if count == 0:
        print(f"     ⚠️  No chunks! Run: python main.py ingest")


def interactive():
    """
    Interactive mode — ask multiple questions without restarting.
    
    WHY interactive mode? In the CLI, launching the Python interpreter +
    loading Ollama connection takes 2-3 seconds each time.
    Interactive mode loads once, then you ask unlimited questions instantly.
    """
    print("\n" + "=" * 60)
    print("  💬 INTERACTIVE MODE — Chat with your documents")
    print("  Type your question and press Enter.")
    print("  Commands: /quit to exit, /stats for stats, /ingest to reload")
    print("=" * 60)
    
    embedder = Embedder()
    store = VectorStore(db_path=CHROMA_PATH)
    retriever = Retriever(embedder, store, top_k=5)
    generator = AnswerGenerator()
    
    if store.collection.count() == 0:
        print("\n  ⚠️  No documents ingested! Run: python main.py ingest")
        return
    
    print(f"\n  📚 {store.collection.count()} chunks loaded. Ask anything!\n")
    
    while True:
        try:
            question = input("  ❓ > ").strip()
            if not question:
                continue
            if question.lower() in ["/quit", "/exit", "/q"]:
                print("  👋 Goodbye!")
                break
            elif question.lower() == "/stats":
                stats()
                continue
            elif question.lower() == "/ingest":
                ingest()
                continue
            
            ask(question)
            print()  # Blank line between Q&A
        except KeyboardInterrupt:
            print("\n  👋 Goodbye!")
            break


def show_help():
    """Show usage instructions."""
    print("""
  📚 RAG Chat with Documents — Usage:
  
  python main.py ingest        Load & embed documents from documents/ folder
  python main.py ask "query"   Ask a single question
  python main.py interactive   Interactive Q&A mode (recommended)
  python main.py stats          Show vector store statistics
  python main.py help           Show this help
  
  Workflow:
    1. Put PDF/TXT/MD files in the documents/ folder
    2. Run: python main.py ingest
    3. Run: python main.py interactive
    4. Ask questions!
""")


if __name__ == "__main__":
    # WHY sys.argv? Simple CLI argument parsing without needing click/argparse.
    # Keeps it student-friendly — no extra dependencies, easy to read.
    if len(sys.argv) < 2:
        show_help()
    elif sys.argv[1] == "ingest":
        ingest()
    elif sys.argv[1] == "ask":
        if len(sys.argv) < 3:
            print("  ❌ Please provide a question: python main.py ask \"your question\"")
        else:
            ask(sys.argv[2])
    elif sys.argv[1] == "interactive":
        interactive()
    elif sys.argv[1] == "stats":
        stats()
    elif sys.argv[1] == "help":
        show_help()
    else:
        show_help()