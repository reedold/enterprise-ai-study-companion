"""HTTP endpoints. Handlers are sync `def` so FastAPI runs the blocking
PDF/embedding/DB/LLM work in its threadpool instead of the event loop."""

import re

from fastapi import APIRouter, File, Form, UploadFile, status

from app.core.config import get_settings
from app.core.exceptions import InvalidInputError, PayloadTooLargeError
from app.models.schemas import StudyMaterialRequest, StudyMaterialResponse, UploadResponse
from app.services.pdf_service import ingest_pdf
from app.services.rag_service import generate_study_material

router = APIRouter(prefix="/api/v1", tags=["knowledge-engine"])

_ID_RE = re.compile(r"^[\w\-. ]{1,128}$")


def _read_pdf(file: UploadFile) -> bytes:
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise PayloadTooLargeError(f"File exceeds {get_settings().max_upload_mb} MB limit")
    if not data.startswith(b"%PDF-"):
        raise InvalidInputError("Uploaded file is not a valid PDF")
    return data


@router.post("/upload-pdf", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def upload_pdf(
    file: UploadFile = File(..., description="PDF document"),
    subject_id: str = Form(..., description="Subject / course identifier"),
    topic_name: str | None = Form(None, description="Optional topic label for all chunks"),
) -> UploadResponse:
    subject_id = subject_id.strip()
    if not _ID_RE.match(subject_id):
        raise InvalidInputError("subject_id must be 1-128 chars of letters, digits, space, _ - .")
    topic_name = (topic_name or "").strip()[:256] or None

    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise InvalidInputError("Only .pdf files are accepted")

    try:
        pdf_bytes = _read_pdf(file)
    finally:
        file.file.close()

    return ingest_pdf(pdf_bytes, filename, subject_id, topic_name)


@router.post("/generate-study-material", response_model=StudyMaterialResponse)
def generate_material(payload: StudyMaterialRequest) -> StudyMaterialResponse:
    return generate_study_material(payload)


@router.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok"}
