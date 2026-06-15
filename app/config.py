"""Application settings, loaded from environment / .env at startup."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Configured providers/models (label, HF id, or local path). Swappable via env.
    mt_model: str = "google-translate-v2"
    asr_model: str = "sudoping01/bambara-asr-v2"

    # Inference device for local models (ASR): cpu | cuda | mps
    device: str = "cpu"

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
