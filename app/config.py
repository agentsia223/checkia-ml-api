"""Application settings, loaded from environment / .env at startup."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ALLOWED_ASR_QUANTIZATION = {"int8", "none"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Configured providers/models (label, HF id, or local path). Swappable via env.
    mt_model: str = "google-translate-v2"
    asr_model: str = "sudoping01/bambara-asr-v2"

    # Inference device for local models (ASR): cpu | cuda | mps
    device: str = "cpu"

    # Hard cap on ASR decoding length. Whisper defaults to max_length=448;
    # an under-prompted model can ramble to that cap, making every clip cost
    # ~448 sequential decoder steps regardless of audio length.
    asr_max_new_tokens: int = 200

    # Max transcriptions decoded at once (each one is CPU-heavy).
    asr_max_concurrency: int = 2

    # Dynamic int8 quantization of the ASR model's Linear layers. CPU-only;
    # "none" disables it and loads the model at full precision.
    asr_quantization: str = "int8"

    # Pass low_cpu_mem_usage=True to from_pretrained to reduce the peak
    # memory used while loading the model.
    asr_low_cpu_mem: bool = True

    @field_validator("asr_quantization")
    @classmethod
    def _validate_asr_quantization(cls, value: str) -> str:
        if value not in _ALLOWED_ASR_QUANTIZATION:
            raise ValueError(
                f"asr_quantization must be one of {sorted(_ALLOWED_ASR_QUANTIZATION)}, got {value!r}"
            )
        return value

    # Secrets: presence only is ever inspected; never logged.
    hf_token: str | None = None
    google_translate_api_key: str | None = None

    # Timeout (seconds) for the translation provider HTTP call.
    translate_timeout: float = 10.0

    @property
    def registry(self) -> dict[str, str]:
        return {"translation": self.mt_model, "asr": self.asr_model}


@lru_cache
def get_settings() -> Settings:
    return Settings()
