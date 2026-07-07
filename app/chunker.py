"""
MODULE 2: CHUNKER
=================
WHY THIS EXISTS:
  Documents are too big to search as one piece. A 50-page PDF is ~25,000 words.
  If you embed it as one vector, it becomes a "blur" — close to everything, 
  specific to nothing. Search fails.
  
  Chunking breaks documents into small pieces (~500 words each) so each piece
  gets its own vector. Now "revenue" chunks are separate from "hiring" chunks.
  Search becomes precise.

WHAT IT DOES:
  - Takes a document (potentially thousands of words)
  - Splits it into chunks of ~500 characters (configurable)
  - Adds overlap of 50 characters between chunks (so nothing is lost at boundaries)
  - Preserves metadata (source, page) on every chunk

WHY ~500 characters?
  Too small (100): loses context — the chunk doesn't have enough info to answer questions
  Too large (2000): loses precision — the chunk covers too many topics
  500 is the Goldilocks zone: enough context, enough precision
  
  In production, you'd experiment with 250, 500, 1000 and measure retrieval quality.

WHY overlap of 50?
  If you cut at exactly 500, you might split a sentence in half:
    Chunk 1: "...the revenue model is based on"
    Chunk 2: "subscription fees charged monthly..."
  The answer is split across the boundary. With overlap, the last 50 chars of 
  chunk 1 appear at the start of chunk 2. Information is preserved.
"""


from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to split
        chunk_size: Target characters per chunk (500 is a good default)
        overlap: Characters of overlap between chunks (prevents losing info at boundaries)
    
    Returns:
        List of text chunks
    
    Example:
        >>> chunks = chunk_text("Hello world " * 100, chunk_size=50, overlap=10)
        >>> len(chunks)  # Many chunks, each ~50 chars, overlapping by 10
    """
    if len(text) <= chunk_size:
        # WHY return the whole thing? If the document is already small enough,
        # no point splitting it. One chunk = one vector. Simple.
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Take a slice of chunk_size characters
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # WHY move start by (chunk_size - overlap)?
        # If chunk_size=500 and overlap=50, we move forward 450 characters.
        # The last 50 chars of this chunk = first 50 chars of next chunk.
        # This ensures overlap.
        start = start + (chunk_size - overlap)
    
    return chunks


def chunk_documents(documents: List[Dict], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Chunk a list of documents (from loader.py) into smaller pieces.
    
    WHY preserve metadata on every chunk?
      When the retriever finds chunk #47 is the best match for a question,
      we need to know: which file did it come from? What page?
      Without metadata, we'd have the text but couldn't cite the source.
      "According to page 5 of report.pdf" builds trust. "According to some text" doesn't.
    
    Returns:
        List of chunks, each with:
            - text: the chunk content
            - source: original filename
            - page: page number
            - chunk_index: which chunk this is (0, 1, 2...)
    """
    all_chunks = []
    
    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size, overlap)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_index": i  # WHY? So we can trace back to exact position if needed
            })
    
    return all_chunks


# WHY this test block? So students can run this file standalone and see chunking work.
if __name__ == "__main__":
    print("\n=== Chunker Test ===\n")
    
    # Simulate a document
    sample_text = "This is sentence one. " * 200  # ~3600 chars
    print(f"  Original text: {len(sample_text)} characters")
    
    chunks = chunk_text(sample_text, chunk_size=500, overlap=50)
    print(f"  After chunking: {len(chunks)} chunks")
    print(f"  Chunk 1: {len(chunks[0])} chars — ends with: '...{chunks[0][-30:]}'")
    print(f"  Chunk 2: {len(chunks[1])} chars — starts with: '{chunks[1][:30]}...'")
    print(f"  ⚡ Notice the overlap — last chars of chunk 1 = first chars of chunk 2")