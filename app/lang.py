"""Friendly language codes <-> Google Cloud Translation codes.

The API boundary speaks ``bm`` / ``fr`` / ``en``; the translation backend speaks
Google's language codes. Keeping the mapping here keeps clients simple and the
backend interchangeable.
"""

FRIENDLY_TO_GOOGLE: dict[str, str] = {
    "bm": "bm",
    "fr": "fr",
    "en": "en",
}

SUPPORTED: tuple[str, ...] = tuple(FRIENDLY_TO_GOOGLE)


class UnsupportedLanguageError(ValueError):
    """Raised when a friendly language code is not supported."""


def to_google(code: str) -> str:
    """Map a friendly code (case-insensitive) to its Google language code."""
    try:
        return FRIENDLY_TO_GOOGLE[code.lower()]
    except (KeyError, AttributeError):
        raise UnsupportedLanguageError(
            f"Unsupported language '{code}'. Supported: {', '.join(SUPPORTED)}."
        )
