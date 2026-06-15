from tests.conftest import FAKE_MT


def test_translate_happy_path(client):
    resp = client.post(
        "/translate",
        json={"text": "Bonjour", "source_lang": "fr", "target_lang": "bm"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["translated_text"] == "[fr->bm] Bonjour"
    assert body["source_lang"] == "fr"
    assert body["target_lang"] == "bm"
    assert body["model"] == FAKE_MT


def test_translate_unsupported_source(client):
    resp = client.post(
        "/translate",
        json={"text": "hi", "source_lang": "es", "target_lang": "fr"},
    )
    assert resp.status_code == 422
    assert "Unsupported language" in resp.json()["detail"]


def test_translate_unsupported_target(client):
    resp = client.post(
        "/translate",
        json={"text": "hi", "source_lang": "fr", "target_lang": "de"},
    )
    assert resp.status_code == 422


def test_translate_rejects_empty_text(client):
    resp = client.post(
        "/translate",
        json={"text": "", "source_lang": "fr", "target_lang": "bm"},
    )
    assert resp.status_code == 422  # pydantic min_length


def test_translate_requires_fields(client):
    resp = client.post("/translate", json={"text": "hi"})
    assert resp.status_code == 422
