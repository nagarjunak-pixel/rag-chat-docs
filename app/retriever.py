"""
MODULE 4: RETRIEVER
===================
WHY THIS EXISTS:
  The embedder knows how to convert text → vectors.
  The vector store knows how to search vectors.
  But we need something that ties them together for a USER'S QUESTION.
  
  This module is the "R" in RAG — Retrieval.
  
  Steps:
    1. Take user's question ("How much revenue did we make?")
    2. Convert question to a vector (same embedding model as documents)
    3. Search vector store for closest chunks
    4. Return those chunks with source metadata

WHY use the SAME embedding model for questions and documents?
  If you embed documents with model A and questions with model B,
  the vectors live in different "spaces" — similarity search is meaningless.
  Same model = same space = meaningful similarity.

WHY a separate class instead of just calling embedder + store directly?
  Separation of concerns. The retriever encapsulates the "search" logic.
  Later we can add reranking, query rewriting, or HyDE without touching
  the embedder or vector store. Each module does one thing.
"""

from typing import List, Dict
from embedder import Embedder, VectorStore


class Retriever:
    """Searches the vector store for chunks relevant to a question."""
    
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = 5):
        """
        Args:
            embedder: The Embedder instance (to convert questions to vectors)
            store: The VectorStore instance (to search for similar chunks)
            top_k: How many chunks to retrieve (5 is a good default)
        
        WHY top_k=5?
          Too few (1-2): might miss the right chunk if top match is wrong
          Too many (20+): dilutes LLM context with irrelevant info
          5 is the sweet spot — enough to find the answer, not so much that
          the LLM gets confused by irrelevant context.
        """
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        print(f"  🔍 Retriever ready — top_k: {top_k}")
    
    def retrieve(self, question: str) -> List[Dict]:
        """
        Find the most relevant chunks for a question.
        
        THE FLOW:
          Question text → Embedder.embed() → vector → VectorStore.search() → chunks
          
        This is the core of RAG. Everything else is plumbing around this.
        
        Returns:
            List of retrieved chunks, each with:
                - text: chunk content
                - source: filename
                - page: page number
                - score: similarity score (0-1, higher = better)
        """
        # Step 1: Convert question to vector
        # WHY? The vector store stores document vectors. To find similar documents,
        # we need the question as a vector in the same "space".
        query_embedding = self.embedder.embed(question)
        
        # Step 2: Search vector store
        # This computes cosine similarity between question vector and all document vectors
        # Returns the top_k closest matches
        results = self.store.search(query_embedding, top_k=self.top_k)
        
        return results
    
    def retrieve_with_context(self, question: str) -> str:
        """
        Retrieve chunks and format them as context text for the LLM.
        
        WHY format as text? The LLM doesn't understand Python dicts.
        We need to present the retrieved chunks as readable text that the
        LLM can reference when answering.
        
        Format:
            [Source: report.pdf, page 5]
            The company's revenue model is based on subscription fees...
            
            [Source: notes.txt, page 0]
            Q4 revenue was $2.3M, up 20% year-over-year...
        
        This format makes it easy for the LLM to cite sources in its answer.
        """
        chunks = self.retrieve(question)
        
        if not chunks:
            return "No relevant documents found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['source']}, page {chunk['page']}, "
                f"relevance: {chunk['score']:.2f}]\n{chunk['text']}"
            )
        
        return "\n\n---\n\n".join(context_parts)


# WHY this test block? Students can verify retrieval works before building
# the full LLM answer generation pipeline.
if __name__ == "__main__":
    print("\n=== Retriever Test ===\n")
    
    embedder = Embedder()
    store = VectorStore()
    retriever = Retriever(embedder, store)
    
    # Test with a question
    question = "What is the revenue model?"
    print(f"  ❓ Question: {question}")
    
    results = retriever.retrieve(question)
    print(f"\n  📄 Retrieved {len(results)} chunks:")
    
    for i, r in enumerate(results, 1):
        preview = r["text"][:80].replace("\n", " ")
        print(f"    {i}. [{r['source']}, p.{r['page']}] score={r['score']:.4f}")
        print(f"       {preview}...")
    
    # Show formatted context
    print(f"\n  📝 Formatted context for LLM:")
    context = retriever.retrieve_with_context(question)
    print(f"  {context[:300]}...")