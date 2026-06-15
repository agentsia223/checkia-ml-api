"""Translation service: Google Cloud Translation API (Basic / v2) client.

The translation backend is configured via env. This client speaks the v2 REST
API over httpx and reports the configured label as the served model.
"""

from __future__ import annotations

import httpx

from ..lang import to_google

BASE_URL = "https://translation.googleapis.com/language/translate/v2"


class GoogleTranslationService:
    def __init__(
        self,
        model_label: str,
        api_key: str | None,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model_label = model_label
        self._api_key = api_key
        self._timeout = timeout
        self._base_url = BASE_URL
        self._transport = transport
        self._client: httpx.Client | None = None

    @property
    def model_name(self) -> str:
        return self.model_label

    def load(self) -> "GoogleTranslationService":
        if not self._api_key:
            raise RuntimeError(
                "GOOGLE_TRANSLATE_API_KEY is required for the translation backend"
            )
        self._client = httpx.Client(timeout=self._timeout, transport=self._transport)
        return self

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if self._client is None:
            raise RuntimeError("GoogleTranslationService not loaded")

        src = to_google(source_lang)
        tgt = to_google(target_lang)

        resp = self._client.post(
            self._base_url,
            params={"key": self._api_key},
            data={"q": text, "source": src, "target": tgt, "format": "text"},
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload["data"]["translations"][0]["translatedText"]
