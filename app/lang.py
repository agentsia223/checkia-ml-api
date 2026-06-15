"""Friendly language codes <-> NLLB FLORES-200 codes.

The API boundary speaks ``bm`` / ``fr`` / ``en``; the translation model speaks
NLLB codes. Keeping the mapping here keeps clients simple and the model
interchangeable.
"""

FRIENDLY_TO_NLLB: dict[str, str] = {
    "bm": "bam_Latn",
    "fr": "fra_Latn",
    "en": "eng_Latn",
}

SUPPORTED: tuple[str, ...] = tuple(FRIENDLY_TO_NLLB)


class UnsupportedLanguageError(ValueError):
    """Raised when a friendly language code is not supported."""


def to_nllb(code: str) -> str:
    """Map a friendly code (case-insensitive) to its NLLB code."""
    try:
        return FRIENDLY_TO_NLLB[code.lower()]
    except (KeyError, AttributeError):
        raise UnsupportedLanguageError(
            f"Unsupported language '{code}'. Supported: {', '.join(SUPPORTED)}."
        )
