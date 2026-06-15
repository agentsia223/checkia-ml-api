"""Opt-in tests that download and run the actually-configured models.

Run with:  uv run pytest -m real
These are skipped by default (``-m "not real"``).
"""

import numpy as np
import pytest

from app.config import get_settings

pytestmark = pytest.mark.real


def test_translation_real():
    settings = get_settings()
    from app.services.translation import TranslationService

    service = TranslationService(
        settings.mt_model, settings.device, settings.hf_token
    ).load()
    out = service.translate("Bonjour le monde", "fr", "en")
    assert isinstance(out, str) and out.strip()


def test_asr_real():
    settings = get_settings()
    if not settings.hf_token:
        pytest.skip("HF_TOKEN not set; gated ASR model unavailable")

    from app.services.asr import ASRService

    service = ASRService(settings.asr_model, settings.device, settings.hf_token).load()

    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    samples = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    out = service.transcribe(samples, language="bm")
    assert isinstance(out, str)
