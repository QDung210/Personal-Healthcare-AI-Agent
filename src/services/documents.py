"""
Document RAG Service - PDF Upload & Processing
Handles user-uploaded PDFs with Docling OCR, chunking, embedding, and FAISS storage
"""

import os, re, uuid, json
from typing import List, Optional, Dict
from pathlib import Path
from dataclasses import dataclass
import faiss, numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from src.models.model import EMBEDDER, QDRANT_CLIENT
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from typing import Tuple
from datetime import datetime
from src.config import (
    PDF_MAX_SIZE_MB,
    PDF_STORAGE_PATH,
    FAISS_INDEX_PATH,
    PDF_METADATA_PATH,
    CHUNK_REGEX,
    OCR_LANGS
)

for p in [PDF_STORAGE_PATH, FAISS_INDEX_PATH, PDF_METADATA_PATH]:
    Path(p).mkdir(exist_ok=True)

class PDFChunk(BaseModel):
    chunk_id: str
    chunk_index: int
    header: str
    content: str
    page_number: Optional[int] = None
    char_count: int = 0

class PDFDocument(BaseModel):
    doc_id: str
    session_id: str
    user_id: str
    file_name: str
    file_size: int
    upload_time: str
    total_chunks: int
    total_pages: Optional[int] = None
    faiss_index_path: str
    metadata_path: str
    ocr_status: str = "processing"
    error_message: Optional[str] = None

class DocumentSearchResult(BaseModel):
    chunk_id: str
    content: str
    header: str
    score: float
    chunk_index: int
    page_number: Optional[int] = None


@dataclass
class DocumentDeps:
    """Dependencies for document processing"""
    embedder: SentenceTransformer
    upload_dir: Path
    faiss_dir: Path
    metadata_dir: Path


# Global deps
DOC_DEPS = DocumentDeps(
    embedder=EMBEDDER,
    upload_dir=PDF_STORAGE_PATH,
    faiss_dir=FAISS_INDEX_PATH,
    metadata_dir=PDF_METADATA_PATH
)

def validate_pdf_file(file_path: Path, file_size: int) -> bool:
    if file_size > PDF_MAX_SIZE_MB * 1024 * 1024:
        return False
    if file_path.suffix.lower() != '.pdf':
        return False
    if not file_path.exists():
        return False
    return True

def extract_text_with_docling(pdf_path: Path) -> Tuple[str, int]:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    ocr_options = EasyOcrOptions(force_full_page_ocr=True, lang=OCR_LANGS)
    pipeline_options.ocr_options = ocr_options
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
    result = converter.convert(str(pdf_path))
    text = result.document.export_to_markdown()
    total_pages = len(result.document.pages) if hasattr(result.document, 'pages') else None
    return text, total_pages

def chunk_by_double_hash(text: str) -> List[PDFChunk]:
    pattern = r"^##\s+.*$"
    lines = text.splitlines()
    header_indices = [i for i, line in enumerate(lines) if re.match(pattern, line)]
    chunks = []
    for idx in range(len(header_indices)):
        start = header_indices[idx]
        end = header_indices[idx+1] if idx+1 < len(header_indices) else len(lines)
        chunk_lines = lines[start:end]
        header = lines[start].strip()
        content = '\n'.join(chunk_lines).strip()
        chunk = PDFChunk(
            chunk_id=str(uuid.uuid4()),
            chunk_index=idx,
            header=header,
            content=content,
            char_count=len(content)
        )
        chunks.append(chunk)
    return chunks

def embed_chunks(chunks: List[PDFChunk]) -> np.ndarray:
    texts = [chunk.content for chunk in chunks]
    embeddings = EMBEDDER.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=8)
    return embeddings

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def save_faiss_index(index: faiss.IndexFlatIP, chunks: List[PDFChunk], session_id: str, user_id: str):
    user_dir = Path(FAISS_INDEX_PATH) / user_id
    user_dir.mkdir(exist_ok=True)
    meta_dir = Path(PDF_METADATA_PATH) / user_id
    meta_dir.mkdir(exist_ok=True)
    index_path = user_dir / f"{session_id}.faiss"
    metadata_path = meta_dir / f"{session_id}.json"
    faiss.write_index(index, str(index_path))
    metadata = {
        "session_id": session_id,
        "user_id": user_id,
        "total_chunks": len(chunks),
        "chunks": [chunk.model_dump() for chunk in chunks]
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return index_path, metadata_path

def load_faiss_index(session_id: str, user_id: str):
    index_path = Path(FAISS_INDEX_PATH) / user_id / f"{session_id}.faiss"
    metadata_path = Path(PDF_METADATA_PATH) / user_id / f"{session_id}.json"
    index = faiss.read_index(str(index_path))
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    chunks = [PDFChunk(**chunk_data) for chunk_data in metadata['chunks']]
    return index, chunks

def search_session_documents(session_id: str, user_id: str, query: str, top_k: int = 3) -> List[DocumentSearchResult]:
    index, chunks = load_faiss_index(session_id, user_id)
    query_embedding = EMBEDDER.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            chunk = chunks[idx]
            results.append(DocumentSearchResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                header=chunk.header,
                score=float(score),
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number
            ))
    return results

async def process_uploaded_pdf(file_path: Path, file_size: int, session_id: str, user_id: str, file_name: str) -> PDFDocument:
    doc_id = str(uuid.uuid4())
    if not validate_pdf_file(file_path, file_size):
        return PDFDocument(doc_id=doc_id, session_id=session_id, user_id=user_id, file_name=file_name, file_size=file_size, upload_time=datetime.now().isoformat(), total_chunks=0, faiss_index_path="", metadata_path="", ocr_status="failed", error_message="Invalid PDF")
    text, total_pages = extract_text_with_docling(file_path)
    # Save OCR text to file for inspection
    ocr_text_path = Path(PDF_METADATA_PATH) / user_id / f"{session_id}_ocr.txt"
    ocr_text_path.parent.mkdir(exist_ok=True)
    with open(ocr_text_path, "w", encoding="utf-8") as f:
        f.write(text)
    chunks = chunk_by_double_hash(text)
    if len(chunks) == 0:
        return PDFDocument(doc_id=doc_id, session_id=session_id, user_id=user_id, file_name=file_name, file_size=file_size, upload_time=datetime.now().isoformat(), total_chunks=0, total_pages=total_pages, faiss_index_path="", metadata_path="", ocr_status="failed", error_message="Không tìm thấy section nào (##) trong file PDF")
    texts = [chunk.content for chunk in chunks]
    embeddings = EMBEDDER.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=8)
    points = []
    for i, chunk in enumerate(chunks):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": embeddings[i].tolist(),
            "payload": {
                "header": chunk.header,
                "content": chunk.content,
                "session_id": session_id,
                "user_id": user_id
            }
        })
    if not QDRANT_CLIENT.collection_exists(collection_name="documents"):
        QDRANT_CLIENT.create_collection(
            collection_name="documents",
            vectors_config={"size": embeddings.shape[1], "distance": "Cosine"}
        )
    QDRANT_CLIENT.upsert(collection_name="documents", points=points)
    return PDFDocument(doc_id=doc_id, session_id=session_id, user_id=user_id, file_name=file_name, file_size=file_size, upload_time=datetime.now().isoformat(), total_chunks=len(chunks), total_pages=total_pages, faiss_index_path="", metadata_path="", ocr_status="completed")

def list_user_sessions(user_id: str) -> List[str]:
    user_faiss_dir = Path(FAISS_INDEX_PATH) / user_id
    if not user_faiss_dir.exists():
        return []
    faiss_files = list(user_faiss_dir.glob("*.faiss"))
    session_ids = [f.stem for f in faiss_files]
    return session_ids

def delete_session_documents(session_id: str, user_id: str) -> bool:
    index_path = Path(FAISS_INDEX_PATH) / user_id / f"{session_id}.faiss"
    metadata_path = Path(PDF_METADATA_PATH) / user_id / f"{session_id}.json"
    deleted = False
    if index_path.exists():
        index_path.unlink()
        deleted = True
    if metadata_path.exists():
        metadata_path.unlink()
        deleted = True
    return deleted

def get_session_stats(session_id: str, user_id: str) -> Optional[Dict]:
    metadata_path = Path(PDF_METADATA_PATH) / user_id / f"{session_id}.json"
    if not metadata_path.exists():
        return None
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    chunks = metadata.get('chunks', [])
    return {
        "session_id": session_id,
        "total_chunks": len(chunks),
        "total_chars": sum(c.get('char_count', 0) for c in chunks),
        "headers": [c.get('header', '')[:50] for c in chunks[:5]]
    }
