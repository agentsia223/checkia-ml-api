"""POST /translate"""

from fastapi import APIRouter, HTTPException, Request

from ..lang import UnsupportedLanguageError
from ..schemas import TranslateRequest, TranslateResponse

router = APIRouter()


@router.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest, request: Request) -> TranslateResponse:
    service = request.app.state.translation
    try:
        translated = service.translate(req.text, req.source_lang, req.target_lang)
    except UnsupportedLanguageError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # model/inference failure
        raise HTTPException(status_code=503, detail=f"Translation failed: {exc}")

    return TranslateResponse(
        translated_text=translated,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        model=service.model_name,
    )
