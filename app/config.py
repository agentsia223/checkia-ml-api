"""Application settings, loaded from environment / .env at startup."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Configured models (Hugging Face id or local path). Swappable via env.
    mt_model: str = "facebook/nllb-200-distilled-600M"
    asr_model: str = "sudoping01/bambara-asr-v2"

    # Inference device: cpu | cuda | mps
    device: str = "cpu"

    # Token for gated models. Presence only is ever inspected; never logged.
    hf_token: str | None = None

    @property
    def registry(self) -> dict[str, str]:
        return {"translation": self.mt_model, "asr": self.asr_model}


@lru_cache
def get_settings() -> Settings:
    return Settings()
