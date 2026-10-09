"""Pydantic request / response models."""

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------- upload ---
class UploadResponse(BaseModel):
    document_id: str
    filename: str
    subject_id: str
    topic_name: str | None
    pages: int
    ocr_pages: int = 0
    chunks_stored: int


# ----------------------------------------------------- study material in ---
class StudyMaterialRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Topic or question")
    subject_id: str | None = Field(None, max_length=128)
    topic_name: str | None = Field(None, max_length=256)
    top_k: int | None = Field(None, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("query must contain at least 3 non-whitespace characters")
        return v


# ---------------------------------------- LLM output (validated strictly) ---
class KeyConcept(BaseModel):
    term: str = Field(..., min_length=1)
    explanation: str = Field(..., min_length=1)
    sources: list[int] = Field(default_factory=list)


class Formula(BaseModel):
    name: str = Field(..., min_length=1)
    expression: str = Field(..., min_length=1)
    explanation: str = ""
    sources: list[int] = Field(default_factory=list)


class SelfAssessmentQuestion(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    sources: list[int] = Field(default_factory=list)


class LLMStudyMaterial(BaseModel):
    """Exact JSON shape the LLM must return."""

    insufficient_context: bool = False
    summary: str = ""
    summary_sources: list[int] = Field(default_factory=list)
    key_concepts: list[KeyConcept] = Field(default_factory=list)
    formulas: list[Formula] = Field(default_factory=list)
    self_assessment: list[SelfAssessmentQuestion] = Field(default_factory=list)


# ---------------------------------------------------- study material out ---
class SourceChunk(BaseModel):
    ref: int
    chunk_id: str
    topic_name: str | None
    filename: str | None
    page: int | None
    similarity: float
    excerpt: str


class StudyMaterialResponse(BaseModel):
    query: str
    grounded: bool = Field(..., description="False when documents did not cover the topic")
    message: str | None = None
    summary: str
    summary_sources: list[int]
    key_concepts: list[KeyConcept]
    formulas: list[Formula]
    self_assessment: list[SelfAssessmentQuestion]
    sources: list[SourceChunk]
    model: str
