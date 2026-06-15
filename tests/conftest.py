"""Test fixtures: an app wired with fake services (no model weights downloaded)."""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.config import Settings
from app.lang import to_nllb
from app.main import _noop_lifespan, create_app

FAKE_MT = "fake-mt-model"
FAKE_ASR = "fake-asr-model"


class FakeTranslationService:
    model_name = FAKE_MT

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        # Exercise the real language mapping so unsupported codes still raise.
        to_nllb(source_lang)
        to_nllb(target_lang)
        return f"[{source_lang}->{target_lang}] {text}"


class FakeASRService:
    model_name = FAKE_ASR

    def transcribe(self, samples, language=None) -> str:
        return "fake transcription"


def make_wav_bytes(seconds: float = 1.0, sr: int = 16_000, freq: float = 440.0) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    samples = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav_bytes()


@pytest.fixture
def client():
    app = create_app(lifespan=_noop_lifespan)
    app.state.settings = Settings(
        _env_file=None, mt_model=FAKE_MT, asr_model=FAKE_ASR, device="cpu"
    )
    app.state.translation = FakeTranslationService()
    app.state.asr = FakeASRService()
    with TestClient(app) as test_client:
        yield test_client
