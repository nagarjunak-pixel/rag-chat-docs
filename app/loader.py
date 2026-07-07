"""
MODULE 1: DOCUMENT LOADER
=========================
WHY THIS EXISTS:
  You can't search documents you haven't read into memory.
  This module reads PDF and text files from a folder and returns their text content.
  
WHAT IT DOES:
  - Scans the documents/ folder for .pdf and .txt files
  - Extracts text from each file using PyMuPDF (for PDFs) or plain read (for text)
  - Returns a list of documents with metadata (filename, page number)
  
WHY PyMuPDF (fitz)?
  PDFs are binary files with complex formatting. You can't just open() them.
  PyMuPDF is a fast, reliable library that extracts clean text from PDFs.
  It also tells us page numbers, which we need for citations later.

WHY keep metadata (filename, page)?
  When the LLM answers a question, we want to say "according to page 5 of report.pdf"
  Without metadata, we'd have the text but no way to trace it back to the source.
  Citations = trust. Users need to verify answers came from real sources.
"""

import fitz  # PyMuPDF — extracts text from PDF files
from pathlib import Path
from typing import List, Dict


def load_documents(folder_path: str) -> List[Dict]:
    """
    Load all PDF and text files from a folder.
    
    Args:
        folder_path: Path to folder containing documents
    
    Returns:
        List of dicts, each with:
            - text: the extracted text content
            - source: filename (e.g. "report.pdf")
            - page: page number (for PDFs) or 0 (for text files)
    
    Example return:
        [
            {"text": "Chapter 1: Introduction...", "source": "report.pdf", "page": 1},
            {"text": "Chapter 2: Methods...", "source": "report.pdf", "page": 2},
            {"text": "Meeting notes from...", "source": "notes.txt", "page": 0},
        ]
    """
    folder = Path(folder_path)
    documents = []
    
    if not folder.exists():
        print(f"  ⚠️  Folder {folder_path} does not exist")
        return documents
    
    # WHY .pdf and .txt? These are the most common document formats.
    # You can add .docx, .html, .md later — same pattern, different parser.
    files = list(folder.glob("*.pdf")) + list(folder.glob("*.txt")) + list(folder.glob("*.md"))
    
    if not files:
        print(f"  ⚠️  No documents found in {folder_path}")
        print(f"     Add .pdf, .txt, or .md files to that folder")
        return documents
    
    print(f"  📂 Found {len(files)} document(s) in {folder_path}")
    
    for filepath in sorted(files):
        if filepath.suffix.lower() == ".pdf":
            # WHY page-by-page? So we can cite the exact page later.
            # If we dumped all pages into one string, we'd lose page-level citations.
            doc = fitz.open(filepath)
            page_count = len(doc)  # WHY store before close? Can't access len(doc) after doc.close()
            for page_num in range(page_count):
                text = doc[page_num].get_text()
                if text.strip():  # Skip empty pages
                    documents.append({
                        "text": text.strip(),
                        "source": filepath.name,
                        "page": page_num + 1  # WHY +1? Pages are 1-indexed for humans
                    })
            doc.close()
            print(f"     ✓ {filepath.name} — {page_count} page(s)")
        else:
            # Text files: read the whole thing as one "page"
            text = filepath.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                documents.append({
                    "text": text.strip(),
                    "source": filepath.name,
                    "page": 0  # 0 means "whole file, no page number"
                })
                print(f"     ✓ {filepath.name} — text file")
    
    return documents


# WHY this test block? So students can run `python loader.py` and see it work
# without needing the rest of the system.
if __name__ == "__main__":
    print("\n=== Document Loader Test ===\n")
    docs = load_documents("../documents")
    print(f"\n  Total chunks loaded: {len(docs)}")
    for d in docs[:3]:  # Show first 3 as preview
        preview = d["text"][:100].replace("\n", " ")
        print(f"  📄 {d['source']} (page {d['page']}): {preview}...")