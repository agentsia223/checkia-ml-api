from app.config import Settings


def test_settings_translation_defaults():
    s = Settings(_env_file=None)
    assert s.mt_model == "google-translate-v2"
    assert s.translate_timeout == 10.0


def test_settings_accepts_translate_key():
    s = Settings(_env_file=None, google_translate_api_key="abc")
    assert s.google_translate_api_key == "abc"
