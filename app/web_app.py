"""
MODULE 7: FASTAPI WEB APP
=========================
WHY THIS EXISTS:
  The CLI works, but a browser interface makes RAG accessible to non-technical
  users and is better for demos. Students can share a URL instead of explaining
  how to run Python commands.

WHAT IT DOES:
  - Serves a beautiful web UI at http://localhost:8000
  - /api/ask endpoint — takes a question, returns answer with citations
  - /api/ingest endpoint — reloads documents from the documents/ folder
  - /api/stats endpoint — shows how many chunks are stored
  - Streams answers in real-time (user sees words appear as LLM generates)

WHY FastAPI?
  - Fast, async, modern Python web framework
  - Built-in interactive API docs at /docs (great for students)
  - We already have it installed
  - Standard in the AI engineering world

WHY a single HTML file instead of React?
  - Zero build step — no npm, no webpack, no node_modules
  - Students can read and understand the entire frontend in one file
  - Easy to modify — change colors, layout, text in one place
  - FastAPI serves it directly with HTMLResponse

HOW TO RUN:
  python web_app.py
  Then open: http://localhost:8000
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from loader import load_documents
from chunker import chunk_documents
from embedder import Embedder, VectorStore
from retriever import Retriever
from generator import AnswerGenerator

# WHY these paths? Same as main.py — shared config
DOCS_PATH = os.path.join(os.path.dirname(__file__), "..", "documents")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# WHY initialize once at startup? Embedding model + Chroma connection take ~2 seconds.
# If we did this per request, every question would have 2s overhead.
# This way, the first request is fast, and all subsequent requests are instant.
print("\n  🚀 Starting RAG Web App...")
print("  Initializing components (one-time setup)...")
embedder = Embedder()
store = VectorStore(db_path=CHROMA_PATH)
retriever = Retriever(embedder, store, top_k=5)
generator = AnswerGenerator()
print("  ✅ All components ready!\n")

app = FastAPI(title="RAG Chat with Documents", version="1.0.0")


class QuestionRequest(BaseModel):
    """WHY Pydantic? FastAPI uses Pydantic models for request validation.
    If the user sends bad data, FastAPI returns a clear 422 error automatically."""
    question: str


class IngestRequest(BaseModel):
    """Request model for the ingest endpoint."""
    clear_existing: bool = True


@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Serve the main web UI.
    
    WHY inline HTML? One file = easy for students to understand and modify.
    In production you'd use templates (Jinja2) or a separate frontend (React/Vue).
    For a teaching project, one file is better.
    """
    return HTML_TEMPLATE


@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    """
    Answer a question using RAG.
    
    FLOW:
      1. User sends {"question": "What is the revenue?"}
      2. We embed the question → search Chroma → get relevant chunks
      3. We send question + chunks to GLM-5.2 → get answer with citations
      4. Return {"answer": "...", "sources": [...], "chunks_found": 5}
    
    WHY not stream here? The /api/ask endpoint returns the full answer as JSON.
    For streaming, use /api/ask/stream (below).
    """
    question = request.question
    if not question.strip():
        return {"error": "Question cannot be empty"}
    
    if store.collection.count() == 0:
        return {"error": "No documents ingested. POST to /api/ingest first."}
    
    # Retrieve relevant chunks
    context = retriever.retrieve_with_context(question)
    results = retriever.retrieve(question)
    
    # Generate answer
    answer = generator.generate(question, context)
    
    # Format sources for the frontend
    sources = [{"source": r["source"], "page": r["page"], "score": r["score"]} for r in results]
    
    return {
        "answer": answer,
        "sources": sources,
        "chunks_found": len(results),
        "chunks_in_store": store.collection.count()
    }


@app.post("/api/ask/stream")
async def ask_question_stream(request: QuestionRequest):
    """
    Stream the answer word-by-word.
    
    WHY streaming? User sees the answer appear as it's generated.
    Much better UX than waiting 5 seconds for the full answer.
    
    WHY StreamingResponse? FastAPI's built-in support for streaming responses.
    We yield chunks as the LLM produces them.
    """
    question = request.question
    if not question.strip():
        return {"error": "Question cannot be empty"}
    
    context = retriever.retrieve_with_context(question)
    
    def generate():
        try:
            for chunk in generator.generate_stream(question, context):
                # WHY JSON lines format? Easy to parse in JavaScript with EventSource or fetch.
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/ingest")
async def ingest_documents(request: IngestRequest):
    """
    Reload documents from the documents/ folder.
    
    WHY an ingest endpoint? So users can add new PDFs and reload
    without restarting the server or touching the terminal.
    """
    documents = load_documents(DOCS_PATH)
    if not documents:
        return {"error": "No documents found in documents/ folder"}
    
    chunks = chunk_documents(documents, chunk_size=500, overlap=50)
    
    # Embed all chunks
    embeddings = []
    for chunk in chunks:
        emb = embedder.embed(chunk["text"])
        embeddings.append(emb)
    
    if request.clear_existing:
        store.clear()
    store.add_chunks(chunks, embeddings)
    
    return {
        "status": "success",
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "total_in_store": store.collection.count()
    }


@app.get("/api/stats")
async def get_stats():
    """Show vector store statistics."""
    return {
        "total_chunks": store.collection.count(),
        "embedding_model": "nomic-embed-text",
        "llm_model": "glm-5.2:cloud",
        "documents_folder": os.path.abspath(DOCS_PATH),
        "chroma_path": os.path.abspath(CHROMA_PATH)
    }


# WHY this HTML template? A complete, beautiful UI in one file.
# Students can change colors, layout, text — all in one place.
# Uses modern CSS (flexbox, gradients, animations) — no frameworks needed.
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Chat with Documents</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f0e17 0%, #1a1a2e 100%);
            color: #e0e0e0; min-height: 100vh; padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        .header {
            text-align: center; padding: 30px 0; margin-bottom: 20px;
        }
        .header h1 {
            font-size: 28px; background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .header p { color: #888; margin-top: 8px; font-size: 14px; }
        .stats-bar {
            display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap;
        }
        .stat-card {
            background: #1e1e30; padding: 12px 20px; border-radius: 10px;
            border: 1px solid #2a2a40; flex: 1; min-width: 150px;
        }
        .stat-card .label { font-size: 11px; color: #666; text-transform: uppercase; }
        .stat-card .value { font-size: 20px; font-weight: bold; margin-top: 4px; }
        .chat-box {
            background: #1a1a2e; border-radius: 15px; border: 1px solid #2a2a40;
            padding: 25px; min-height: 400px; margin-bottom: 20px;
        }
        .messages { max-height: 500px; overflow-y: auto; padding-right: 10px; }
        .messages::-webkit-scrollbar { width: 6px; }
        .messages::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
        .msg { margin-bottom: 20px; }
        .msg.user {
            text-align: right;
        }
        .msg.user .bubble {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white; padding: 12px 18px; border-radius: 18px 18px 4px 18px;
            display: inline-block; max-width: 70%; text-align: left;
        }
        .msg.bot .bubble {
            background: #1e1e30; border: 1px solid #2a2a40;
            padding: 15px 20px; border-radius: 18px 18px 18px 4px;
            display: inline-block; max-width: 85%; text-align: left;
            line-height: 1.6; white-space: pre-wrap;
        }
        .sources {
            margin-top: 10px; font-size: 12px; color: #666;
        }
        .source-tag {
            display: inline-block; background: #2a2a40; padding: 2px 10px;
            border-radius: 10px; margin-right: 5px; margin-bottom: 3px;
            color: #aaa; font-size: 11px;
        }
        .input-area {
            display: flex; gap: 10px;
        }
        .input-area input {
            flex: 1; padding: 15px 20px; border-radius: 25px;
            border: 1px solid #2a2a40; background: #1a1a2e; color: #e0e0e0;
            font-size: 15px; outline: none; transition: border-color 0.3s;
        }
        .input-area input:focus { border-color: #667eea; }
        .input-area button {
            padding: 15px 30px; border-radius: 25px; border: none;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white; font-size: 15px; cursor: pointer; transition: opacity 0.3s;
        }
        .input-area button:hover { opacity: 0.9; }
        .input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
        .actions {
            display: flex; gap: 10px; margin-bottom: 15px;
        }
        .actions button {
            padding: 8px 16px; border-radius: 8px; border: 1px solid #2a2a40;
            background: #1e1e30; color: #aaa; cursor: pointer; font-size: 13px;
        }
        .actions button:hover { background: #2a2a40; }
        .typing { color: #666; font-style: italic; }
        .loading-dots span {
            display: inline-block; animation: blink 1.4s infinite;
        }
        .loading-dots span:nth-child(2) { animation-delay: 0.2s; }
        .loading-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes blink { 0%, 80%, 100% { opacity: 0.2; } 40% { opacity: 1; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 RAG Chat with Documents</h1>
            <p>Ask questions about your documents — answers with citations</p>
        </div>
        
        <div class="stats-bar" id="statsBar">
            <div class="stat-card"><div class="label">Chunks Stored</div><div class="value" id="chunkCount">—</div></div>
            <div class="stat-card"><div class="label">Embedding Model</div><div class="value" style="font-size:14px">nomic-embed-text</div></div>
            <div class="stat-card"><div class="label">LLM</div><div class="value" style="font-size:14px">GLM-5.2 Cloud</div></div>
        </div>
        
        <div class="actions">
            <button onclick="ingestDocs()">🔄 Reload Documents</button>
            <button onclick="clearChat()">🗑️ Clear Chat</button>
        </div>
        
        <div class="chat-box">
            <div class="messages" id="messages">
                <div class="msg bot">
                    <div class="bubble">
                        👋 Hi! I can answer questions about documents in the <b>documents/</b> folder.
                        <br><br>
                        Try asking: <i>"What is the revenue model?"</i> or <i>"What products are launching in 2026?"</i>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="input-area">
            <input type="text" id="questionInput" placeholder="Ask a question about your documents..."
                   onkeypress="if(event.key==='Enter') askQuestion()">
            <button id="askBtn" onclick="askQuestion()">Ask 🚀</button>
        </div>
    </div>
    
    <script>
        // WHY this JavaScript? Handles the UI interaction with the FastAPI backend.
        // fetch() calls /api/ask and /api/ingest, updates the DOM with responses.
        
        async function loadStats() {
            const res = await fetch('/api/stats');
            const data = await res.json();
            document.getElementById('chunkCount').textContent = data.total_chunks;
        }
        
        function addMessage(role, text, sources) {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = 'msg ' + role;
            
            let html = '<div class="bubble">' + text.replace(/\\n/g, '<br>') + '</div>';
            if (sources && sources.length > 0) {
                html += '<div class="sources"><b>Sources:</b> ';
                sources.forEach(s => {
                    html += `<span class="source-tag">📄 ${s.source} (p.${s.page}, score: ${s.score.toFixed(2)})</span>`;
                });
                html += '</div>';
            }
            div.innerHTML = html;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }
        
        async function askQuestion() {
            const input = document.getElementById('questionInput');
            const btn = document.getElementById('askBtn');
            const question = input.value.trim();
            if (!question) return;
            
            // Show user question
            addMessage('user', question);
            input.value = '';
            btn.disabled = true;
            
            // Show typing indicator
            const messages = document.getElementById('messages');
            const typingDiv = document.createElement('div');
            typingDiv.className = 'msg bot';
            typingDiv.id = 'typing';
            typingDiv.innerHTML = '<div class="bubble typing">Searching documents and generating answer<span class="loading-dots"><span>.</span><span>.</span><span>.</span></span></div>';
            messages.appendChild(typingDiv);
            messages.scrollTop = messages.scrollHeight;
            
            try {
                const res = await fetch('/api/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                });
                const data = await res.json();
                
                // Remove typing indicator
                document.getElementById('typing')?.remove();
                
                if (data.error) {
                    addMessage('bot', '⚠️ ' + data.error);
                } else {
                    addMessage('bot', data.answer, data.sources);
                }
            } catch (e) {
                document.getElementById('typing')?.remove();
                addMessage('bot', '⚠️ Error: ' + e.message);
            }
            
            btn.disabled = false;
            input.focus();
        }
        
        async function ingestDocs() {
            addMessage('bot', '🔄 Reloading documents from folder...');
            try {
                const res = await fetch('/api/ingest', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({clear_existing: true})
                });
                const data = await res.json();
                if (data.error) {
                    addMessage('bot', '⚠️ ' + data.error);
                } else {
                    addMessage('bot', `✅ Loaded ${data.documents_loaded} documents, created ${data.chunks_created} chunks. Total in store: ${data.total_in_store}`);
                    loadStats();
                }
            } catch (e) {
                addMessage('bot', '⚠️ Error: ' + e.message);
            }
        }
        
        function clearChat() {
            document.getElementById('messages').innerHTML = '';
            addMessage('bot', '👋 Chat cleared. Ask me anything!');
        }
        
        // Load stats on startup
        loadStats();
    </script>
</body>
</html>"""


if __name__ == "__main__":
    # WHY __name__ == "__main__"? So we can also import this as a module
    # (for testing) without starting the server.
    import uvicorn
    
    print("\n" + "=" * 60)
    print("  🌐 RAG Chat with Documents — Web App")
    print("  📖 Open: http://localhost:8000")
    print("  📚 API docs: http://localhost:8000/docs")
    print("  🛑 Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)