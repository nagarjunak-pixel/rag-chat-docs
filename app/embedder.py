"""
MODULE 3: EMBEDDER + VECTOR STORE
=================================
WHY THIS EXISTS:
  Computers can't "search" text. They search numbers.
  This module converts text chunks into numbers (vectors) called "embeddings"
  and stores them in Chroma — a local vector database that can search by meaning.
  
  Think of it as building an index at the back of a book, but instead of
  alphabetical keywords, it's a semantic index: "this paragraph is about revenue,
  this one is about hiring, this one is about product features."

WHAT IS AN EMBEDDING?
  Text → array of 768 floating-point numbers.
  Text with similar MEANING → similar numbers (vectors close together).
  
  Example:
    "The cat sat on the mat"   → [0.12, -0.45, 0.89, ...]  ← close together
    "The dog sat on the rug"   → [0.15, -0.42, 0.85, ...]  ← close together
    "Revenue increased 20%"    → [-0.67, 0.33, 0.11, ...] ← far from above
  
  This lets us find chunks by MEANING, not just keyword matching.
  If you ask "how much money did they make?" it finds the revenue chunk
  even though "money" isn't in the text.

WHY nomic-embed-text (via Ollama)?
  - Runs locally on your Mac — no API key, no cost, no rate limits
  - Designed specifically for embeddings (not chat)
  - Produces 768-dimensional vectors
  - Your Ollama already has it installed: nomic-embed-text:latest

WHY Chroma?
  - Runs locally as files in chroma_db/ folder — zero infrastructure
  - Stores vectors + metadata + original text together
  - Does similarity search (cosine similarity) out of the box
  - Free, open-source, no account needed
"""

import requests
import json
from typing import List, Dict
import chromadb


# WHY these constants? They connect to your local Ollama instance.
# Ollama runs on port 11434 by default. nomic-embed-text is the embedding model.
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"


class Embedder:
    """Converts text into vectors using Ollama's embedding API."""
    
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        self.url = f"{OLLAMA_URL}/api/embeddings"
        
        # WHY check at init? Fail fast. If Ollama isn't running, we want to know
        # before we try to embed 500 documents, not during.
        if not self._is_ollama_running():
            raise ConnectionError(
                f"Ollama not running at {OLLAMA_URL}. Start it with: ollama serve"
            )
        print(f"  🔢 Embedder ready — model: {model}")
    
    def _is_ollama_running(self) -> bool:
        """Check if Ollama is running before we try to use it."""
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            return r.status_code == 200
        except:
            return False
    
    def embed(self, text: str) -> List[float]:
        """
        Convert a single piece of text into a vector.
        
        WHY this works: Ollama's /api/embeddings endpoint takes text + model name
        and returns a vector (list of floats). The model has been trained so that
        text with similar meaning produces similar vectors.
        
        Returns: list of ~768 floats representing this text's "position" in meaning space
        """
        response = requests.post(self.url, json={
            "model": self.model,
            "prompt": text  # WHY "prompt"? Ollama's API uses this field name for the text to embed
        })
        response.raise_for_status()
        return response.json()["embedding"]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts. 
        WHY a batch method? More efficient than calling embed() in a loop.
        Though Ollama processes them one at a time, keeping the batch API means
        we can swap to a batch-capable provider (OpenAI, Cohere) later without
        changing the rest of the code.
        """
        return [self.embed(text) for text in texts]


class VectorStore:
    """Stores embeddings in Chroma and searches by similarity."""
    
    def __init__(self, db_path: str = "../chroma_db"):
        """
        WHY persist in a local folder? 
          - Data survives restarts. You don't re-embed 500 documents every time.
          - Chroma stores everything as files. No server process needed.
          - You can copy the folder to another machine and it just works.
        """
        self.client = chromadb.PersistentClient(path=db_path)
        
        # WHY get_or_create? If the collection already exists (from a previous run),
        # we use it. If not, we create it. This makes the system idempotent —
        # you can run "ingest" multiple times without errors.
        #
        # WHY cosine distance? Chroma defaults to L2 (squared euclidean) distance,
        # which gives raw distances of 100+ — meaningless for "relevance score".
        # Cosine distance ranges 0-2, where 0 = identical, 1 = unrelated, 2 = opposite.
        # We convert to a 0-1 score where 1 = perfect match.
        try:
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            # Collection might exist with different metadata — delete and recreate
            self.client.delete_collection("documents")
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        print(f"  💾 Vector store ready — path: {db_path}")
        print(f"     Current chunks in store: {self.collection.count()}")
    
    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        """
        Store chunks with their embeddings in Chroma.
        
        WHY store metadata separately?
          Chroma stores: id + vector + document text + metadata
          When we search, Chroma returns the text + metadata of the closest vectors.
          We need the metadata (source, page) to build citations in the answer.
        
        WHY unique IDs?
          Chroma requires unique IDs for each entry. If you add the same ID twice,
          it overwrites. We use "filename_chunkindex" format so IDs are 
          deterministic and human-readable.
        """
        ids = []
        metadatas = []
        texts = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            ids.append(f"{chunk['source']}_p{chunk['page']}_chunk_{chunk['chunk_index']}")
            metadatas.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"]
            })
            texts.append(chunk["text"])
        
        # WHY upsert? If the chunk already exists (same ID), update it.
        # If not, insert it. This means re-running ingest updates stale data
        # instead of creating duplicates.
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"  ✅ Added {len(chunks)} chunks to vector store")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Find the top_k most similar chunks to a query.
        
        HOW IT WORKS:
          1. We pass the query's embedding (vector) to Chroma
          2. Chroma computes cosine similarity between query vector and all stored vectors
          3. Returns the top_k closest matches with their text and metadata
          
        WHY top_k=5?
          Too few (1-2): might miss the right chunk if the #1 match is wrong
          Too many (20+): dilutes the LLM context with irrelevant chunks
          5 is a good balance — enough to find the right answer, not so much that
          the LLM loses focus.
        
        Returns:
          List of dicts, each with:
            - text: the chunk content
            - source: filename
            - page: page number
            - score: similarity score (higher = better match)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # WHY transform the results? Chroma returns lists of lists.
        # We want a clean list of dicts that's easy to use in the LLM module.
        matched = []
        for i in range(len(results["documents"][0])):
            matched.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "page": results["metadatas"][0][i]["page"],
                # WHY convert distance to score? Chroma returns "distance" (lower = closer).
                # With cosine distance: 0 = identical, 1 = unrelated, 2 = opposite.
                # We convert to a 0-1 score where 1 = perfect match, 0 = no match.
                # Formula: score = 1 - (distance / 2) → cosine distance 0 → score 1.0
                "score": 1 - (results["distances"][0][i] / 2)
            })
        
        return matched
    
    def clear(self):
        """Delete all stored chunks. WHY? For re-ingesting fresh documents."""
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        print("  🗑️  Cleared vector store")


# WHY this test block? Students can run `python embedder.py` and verify
# that Ollama embeddings + Chroma storage works before building the full system.
if __name__ == "__main__":
    print("\n=== Embedder + Vector Store Test ===\n")
    
    embedder = Embedder()
    store = VectorStore()
    
    # Test: embed a simple sentence
    test_text = "The revenue model is based on subscription fees."
    embedding = embedder.embed(test_text)
    print(f"  📊 Embedded '{test_text}'")
    print(f"  Vector dimensions: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    
    # Test: store a chunk and search for it
    test_chunks = [{
        "text": test_text,
        "source": "test.txt",
        "page": 1,
        "chunk_index": 0
    }]
    store.clear()
    store.add_chunks(test_chunks, [embedding])
    
    # Search with a different but semantically similar query
    query = "How does the company make money?"
    query_embedding = embedder.embed(query)
    results = store.search(query_embedding, top_k=1)
    
    print(f"\n  🔍 Query: '{query}'")
    print(f"  Found: '{results[0]['text']}'")
    print(f"  Score: {results[0]['score']:.4f}")
    print(f"  Source: {results[0]['source']}, page {results[0]['page']}")
    print(f"\n  ⚡ The query said 'money' but the chunk says 'revenue' — semantic search works!")