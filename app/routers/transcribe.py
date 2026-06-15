"""POST /transcribe — accepts a multipart file upload OR a JSON {"audio_url": ...}."""

import httpx
from fastapi import APIRouter, HTTPException, Request

from ..audio import fetch_audio, load_audio
from ..schemas import TranscribeResponse

router = APIRouter()


async def _extract_input(request: Request) -> tuple[bytes | None, str | None, str | None]:
    """Return (file_bytes, audio_url, language) from a JSON or multipart request."""
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="JSON body must be an object.")
        return None, body.get("audio_url"), body.get("language")

    form = await request.form()
    language = form.get("language")
    audio_url = form.get("audio_url")
    upload = form.get("file")
    file_bytes = None
    if upload is not None and hasattr(upload, "read"):
        file_bytes = await upload.read()
        if not file_bytes:
            file_bytes = None
    return file_bytes, audio_url, language


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: Request) -> TranscribeResponse:
    file_bytes, audio_url, language = await _extract_input(request)

    if (file_bytes is None) == (audio_url is None):
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of 'file' (multipart) or 'audio_url' (JSON).",
        )

    if audio_url is not None:
        try:
            file_bytes = await fetch_audio(audio_url)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=422, detail=f"Could not fetch audio_url: {exc}")

    try:
        samples, duration_s = load_audio(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {exc}")

    service = request.app.state.asr
    try:
        text = service.transcribe(samples, language=language)
    except Exception as exc:  # model/inference failure
        raise HTTPException(status_code=503, detail=f"Transcription failed: {exc}")

    return TranscribeResponse(
        text=text,
        language=language,
        duration_s=round(duration_s, 2),
        model=service.model_name,
    )
