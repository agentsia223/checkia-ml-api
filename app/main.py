"""FastAPI application: loads models once at startup, mounts routers."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from .config import Settings, get_settings
from .routers import transcribe, translate
from .schemas import HealthResponse, ModelsResponse
from .services.asr import ASRService
from .services.translation import TranslationService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load configured models once and hold them on app.state."""
    settings = get_settings()
    app.state.settings = settings
    app.state.translation = TranslationService(
        settings.mt_model, settings.device, settings.hf_token
    ).load()
    app.state.asr = ASRService(
        settings.asr_model, settings.device, settings.hf_token
    ).load()
    yield


@asynccontextmanager
async def _noop_lifespan(app: FastAPI):
    """Lifespan that loads nothing — used by tests that inject fake services."""
    yield


def create_app(*, lifespan=lifespan) -> FastAPI:
    app = FastAPI(title="checkia-ml-api", version="0.1.0", lifespan=lifespan)
    app.include_router(translate.router)
    app.include_router(transcribe.router)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/models", response_model=ModelsResponse)
    def models(request: Request) -> ModelsResponse:
        settings: Settings = request.app.state.settings
        return ModelsResponse(translation=settings.mt_model, asr=settings.asr_model)

    return app


app = create_app()
