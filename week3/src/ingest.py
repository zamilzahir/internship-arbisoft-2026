"""
Loads PDFs from data/pdfs/, extracts text, and splits into overlapping
chunks suitable for embedding + retrieval.
"""
from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader

PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"

CHUNK_SIZE = 400       # characters per chunk (kept small since our docs are short)
CHUNK_OVERLAP = 60     # character overlap between consecutive chunks


@dataclass
class Chunk:
    id: str
    text: str
    source: str      # originating filename
    chunk_index: int


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    # Normalize whitespace so chunk boundaries aren't skewed by PDF layout artifacts.
    cleaned = " ".join(text.split())
    chunks = []
    start = 0
    idx = 0
    while start < len(cleaned):
        end = start + chunk_size
        piece = cleaned[start:end]
        if piece.strip():
            chunks.append(Chunk(
                id=f"{source}::chunk{idx}",
                text=piece,
                source=source,
                chunk_index=idx,
            ))
            idx += 1
        start += chunk_size - overlap
    return chunks


def load_and_chunk_all(pdf_dir: Path = PDF_DIR) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")
    for pdf_path in pdf_paths:
        text = extract_text(pdf_path)
        chunks = chunk_text(text, source=pdf_path.name)
        all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_all()
    print(f"Loaded {len(chunks)} chunks from {PDF_DIR}")
    for c in chunks[:3]:
        print(f"\n[{c.id}]\n{c.text[:200]}...")
