"""PDF ingestion: extract -> clean -> chunk -> embed -> store in Supabase."""

import logging
import re
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass

import pymupdf

from app.core.clients import embed_texts, get_supabase
from app.core.config import get_settings
from app.core.exceptions import InvalidInputError, NoContentError, UpstreamServiceError
from app.models.schemas import UploadResponse
from app.services import ocr_service

logger = logging.getLogger(__name__)

# Lines consisting only of a page number: "12", "- 12 -", "Page 3", "3 of 10", "3/10"
_PAGE_NUMBER_RE = re.compile(r"^\W*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\W*$", re.IGNORECASE)
_URL_ONLY_RE = re.compile(r"^\s*(https?://|www\.)\S+\s*$", re.IGNORECASE)
_DOT_LEADER_RE = re.compile(r"\.{4,}\s*\d*\s*$")  # table-of-contents "Intro ........ 3"
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b-\u200f\ufeff]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9\"'(\[])")
_MIN_CHUNK_CHARS = 40
_MIN_TEXT_CHARS_PER_PAGE = 20  # below this, an image-bearing page is treated as scanned
_EDGE_LINES = 3  # lines at top/bottom of a page considered for header/footer detection


@dataclass
class Chunk:
    content: str
    page: int
    index: int


# ------------------------------------------------------------- extraction ---
def _needs_ocr(page: pymupdf.Page, text: str) -> bool:
    """A page with almost no text layer but with images is most likely scanned."""
    return len(re.findall(r"\w", text)) < _MIN_TEXT_CHARS_PER_PAGE and bool(page.get_images())


def extract_pages(pdf_bytes: bytes) -> tuple[list[str], int]:
    """Return (raw text per page, number of pages that were OCR'd)."""
    settings = get_settings()
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises various internal error types
        raise InvalidInputError(f"Could not parse PDF: {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise InvalidInputError("Password-protected PDFs are not supported")
        texts: list[str] = []
        ocr_pages = 0
        for page in doc:
            # sort=True yields natural reading order for multi-column layouts
            text = page.get_text("text", sort=True)
            if (
                _needs_ocr(page, text)
                and ocr_pages < settings.ocr_max_pages
                and ocr_service.ocr_available()
            ):
                try:
                    text = ocr_service.ocr_page(page)
                    ocr_pages += 1
                except Exception:
                    logger.exception("OCR failed on page %d", page.number + 1)
            texts.append(text)
    return texts, ocr_pages


# --------------------------------------------------------------- cleaning ---
def _normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)
    line = _CONTROL_RE.sub("", line)
    return re.sub(r"[ \t]+", " ", line).strip()


def _mask(line: str) -> str:
    # digits are masked so "Chapter 2 - p. 14" and "... p. 15" count as the same footer
    return re.sub(r"\d+", "#", line.lower())


def _edge_lines(lines: list[str], n: int = _EDGE_LINES) -> list[str]:
    non_empty = [ln for ln in lines if ln]
    return non_empty[:n] + non_empty[-n:]


def _repeated_lines(pages_lines: list[list[str]]) -> set[str]:
    """Detect running headers/footers: short lines at the top/bottom of >= 50% of pages."""
    if len(pages_lines) < 3:
        return set()
    counts = Counter()
    for lines in pages_lines:
        counts.update({_mask(ln) for ln in _edge_lines(lines) if len(ln) <= 120})
    threshold = max(2, int(len(pages_lines) * 0.5))
    return {ln for ln, n in counts.items() if n >= threshold}


def clean_pages(raw_pages: list[str]) -> list[str]:
    """Strip noise (headers/footers, page numbers, TOC leaders, URLs) and rebuild paragraphs."""
    pages_lines = [[_normalize_line(ln) for ln in page.splitlines()] for page in raw_pages]
    boilerplate = _repeated_lines(pages_lines)

    cleaned: list[str] = []
    for lines in pages_lines:
        edges = set(_edge_lines(lines))
        kept: list[str] = []
        for ln in lines:
            if not ln:
                kept.append("")  # preserve paragraph break
                continue
            if (
                (ln in edges and _mask(ln) in boilerplate)
                or _PAGE_NUMBER_RE.match(ln)
                or _URL_ONLY_RE.match(ln)
                or _DOT_LEADER_RE.search(ln)
                or not re.search(r"[A-Za-z0-9]", ln)  # pure symbols / separators
            ):
                continue
            kept.append(ln)

        text = "\n".join(kept)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # de-hyphenate line breaks
        text = re.sub(r"\n{2,}", "\u2029", text)               # mark paragraph breaks
        text = re.sub(r"\s*\n\s*", " ", text)                  # unwrap soft line breaks
        text = re.sub(r"\s*\u2029\s*", "\n\n", text)
        text = re.sub(r"[ ]{2,}", " ", text).strip()
        cleaned.append(text)
    return cleaned


# --------------------------------------------------------------- chunking ---
def _split_long(text: str, size: int) -> list[str]:
    """Hard-split a single over-long sentence on word boundaries."""
    parts, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > size:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def _units(text: str, size: int) -> list[str]:
    units: list[str] = []
    for para in text.split("\n\n"):
        for sentence in _SENTENCE_SPLIT_RE.split(para.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            units.extend(_split_long(sentence, size) if len(sentence) > size else [sentence])
    return units


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def chunk_pages(pages: list[str], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Sentence-aware greedy packing up to `chunk_size` characters with word-aligned overlap."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for page_no, text in enumerate(pages, start=1):
        if not text:
            continue
        current = ""
        for unit in _units(text, chunk_size):
            candidate = f"{current} {unit}".strip()
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if len(current) >= _MIN_CHUNK_CHARS:
                chunks.append(Chunk(current, page_no, len(chunks)))
            tail = _overlap_tail(current, chunk_overlap)
            current = f"{tail} {unit}".strip() if len(tail) + 1 + len(unit) <= chunk_size else unit
        if len(current) >= _MIN_CHUNK_CHARS:
            chunks.append(Chunk(current, page_no, len(chunks)))
    return chunks


# ---------------------------------------------------------------- storage ---
def _store_chunks(rows: list[dict], batch_size: int, document_id: str) -> None:
    client = get_supabase()
    inserted = 0
    try:
        for start in range(0, len(rows), batch_size):
            client.table("document_chunks").insert(rows[start : start + batch_size]).execute()
            inserted += len(rows[start : start + batch_size])
    except Exception as exc:
        logger.exception("Supabase insert failed after %d rows", inserted)
        if inserted:  # roll back partial upload so a retry doesn't create duplicates
            try:
                client.table("document_chunks").delete().eq(
                    "metadata->>document_id", document_id
                ).execute()
            except Exception:
                logger.exception("Rollback of document %s failed", document_id)
        raise UpstreamServiceError("Failed to store document chunks in vector database") from exc


# ------------------------------------------------------------ entry point ---
def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
    subject_id: str,
    topic_name: str | None = None,
) -> UploadResponse:
    settings = get_settings()

    raw_pages, ocr_pages = extract_pages(pdf_bytes)
    pages = clean_pages(raw_pages)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        hint = (
            "OCR found no readable text either."
            if ocr_pages
            else "The PDF may be image-only and OCR is disabled or unavailable."
        )
        raise NoContentError(f"No extractable text found. {hint}")

    embeddings = embed_texts([c.content for c in chunks])
    document_id = str(uuid.uuid4())
    rows = [
        {
            "subject_id": subject_id,
            "topic_name": topic_name,
            "content": chunk.content,
            "embedding": vector,
            "metadata": {
                "document_id": document_id,
                "filename": filename,
                "page": chunk.page,
                "chunk_index": chunk.index,
                "total_pages": len(raw_pages),
                "char_count": len(chunk.content),
            },
        }
        for chunk, vector in zip(chunks, embeddings)
    ]
    _store_chunks(rows, settings.insert_batch_size, document_id)

    logger.info(
        "Ingested %s: %d pages (%d OCR) -> %d chunks",
        filename, len(raw_pages), ocr_pages, len(rows),
    )
    return UploadResponse(
        document_id=document_id,
        filename=filename,
        subject_id=subject_id,
        topic_name=topic_name,
        pages=len(raw_pages),
        ocr_pages=ocr_pages,
        chunks_stored=len(rows),
    )
