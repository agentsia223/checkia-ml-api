import pytest

from app.lang import SUPPORTED, UnsupportedLanguageError, to_nllb


@pytest.mark.parametrize(
    "friendly,nllb",
    [("bm", "bam_Latn"), ("fr", "fra_Latn"), ("en", "eng_Latn")],
)
def test_to_nllb_maps_known_codes(friendly, nllb):
    assert to_nllb(friendly) == nllb


def test_to_nllb_is_case_insensitive():
    assert to_nllb("BM") == "bam_Latn"


def test_supported_set():
    assert set(SUPPORTED) == {"bm", "fr", "en"}


@pytest.mark.parametrize("bad", ["es", "", "bambara", "xx"])
def test_to_nllb_rejects_unknown(bad):
    with pytest.raises(UnsupportedLanguageError):
        to_nllb(bad)
