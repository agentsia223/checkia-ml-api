"""Request/response models — the stable API contract.

Treat these as append-only: adding optional fields is safe; renaming or removing
fields breaks clients.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class ModelsResponse(BaseModel):
    """The models currently serving each endpoint (configurable via env)."""

    translation: str
    asr: str


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1)
    source_lang: str = Field(description="Source language: bm, fr, or en")
    target_lang: str = Field(description="Target language: bm, fr, or en")


class TranslateResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str
    model: str


class TranscribeUrlRequest(BaseModel):
    audio_url: str
    language: str | None = None


class TranscribeResponse(BaseModel):
    text: str
    language: str | None = None
    duration_s: float
    model: str
