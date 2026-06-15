import httpx
import pytest

from app.lang import UnsupportedLanguageError
from app.services.translation import BASE_URL, GoogleTranslationService


def _service(handler, api_key="test-key"):
    transport = httpx.MockTransport(handler)
    return GoogleTranslationService(
        "google-translate-v2", api_key, transport=transport
    ).load()


def test_translate_parses_translated_text():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["key"] = request.url.params.get("key")
        captured["body"] = dict(httpx.QueryParams(request.content.decode()))
        return httpx.Response(
            200, json={"data": {"translations": [{"translatedText": "bonjour"}]}}
        )

    out = _service(handler).translate("hello", "en", "fr")
    assert out == "bonjour"
    assert captured["key"] == "test-key"
    assert captured["url"].startswith(BASE_URL)
    assert captured["body"] == {
        "q": "hello",
        "source": "en",
        "target": "fr",
        "format": "text",
    }


def test_translate_rejects_unsupported_language():
    def handler(request):
        raise AssertionError("HTTP must not be reached for a bad language code")

    with pytest.raises(UnsupportedLanguageError):
        _service(handler).translate("hi", "es", "fr")


def test_translate_raises_on_api_error():
    def handler(request):
        return httpx.Response(403, json={"error": {"message": "forbidden"}})

    with pytest.raises(httpx.HTTPStatusError):
        _service(handler).translate("hi", "en", "fr")


def test_load_requires_api_key():
    with pytest.raises(RuntimeError):
        GoogleTranslationService("google-translate-v2", None).load()
