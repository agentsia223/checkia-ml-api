import pytest

from app.lang import SUPPORTED, UnsupportedLanguageError, to_google


@pytest.mark.parametrize("friendly,google", [("bm", "bm"), ("fr", "fr"), ("en", "en")])
def test_to_google_maps_known_codes(friendly, google):
    assert to_google(friendly) == google


def test_to_google_is_case_insensitive():
    assert to_google("BM") == "bm"


def test_supported_set():
    assert set(SUPPORTED) == {"bm", "fr", "en"}


@pytest.mark.parametrize("bad", ["es", "", "bambara", "xx"])
def test_to_google_rejects_unknown(bad):
    with pytest.raises(UnsupportedLanguageError):
        to_google(bad)
