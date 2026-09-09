"""
Extraction + chunking. Chunking strategy: fixed-size token windows with
overlap (NOT whole-book-as-one-vector). Rationale documented in
models.py / architecture.md: whole-book embeddings lose retrieval
precision on 500+ page books, and blow past embedding context limits.
"""
import tiktoken
from pypdf import PdfReader
import docx

encoding = tiktoken.get_encoding("cl100k_base")


def extract_text(file_path: str, filename: str) -> tuple[str, int]:
    """Returns (full_text, page_count)."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages), len(pages)
    elif lower.endswith(".docx"):
        doc = docx.Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs)
        # docx has no fixed page count; approximate via word count / 400
        approx_pages = max(1, len(text.split()) // 400)
        return text, approx_pages
    elif lower.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return text, max(1, len(text.split()) // 400)
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Token-aware sliding window chunking."""
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))
