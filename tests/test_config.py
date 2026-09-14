import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_translation_defaults():
    s = Settings(_env_file=None)
    assert s.mt_model == "google-translate-v2"
    assert s.translate_timeout == 10.0


def test_settings_accepts_translate_key():
    s = Settings(_env_file=None, google_translate_api_key="abc")
    assert s.google_translate_api_key == "abc"


def test_settings_asr_quantization_defaults():
    s = Settings(_env_file=None)
    assert s.asr_quantization == "int8"
    assert s.asr_low_cpu_mem is True


def test_settings_asr_quantization_accepts_none():
    s = Settings(_env_file=None, asr_quantization="none")
    assert s.asr_quantization == "none"


def test_settings_asr_quantization_rejects_invalid_value():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, asr_quantization="bogus")
