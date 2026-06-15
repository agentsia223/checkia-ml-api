import app.routers.transcribe as transcribe_router
from tests.conftest import FAKE_ASR


def test_transcribe_file_upload(client, wav_bytes):
    resp = client.post(
        "/transcribe",
        files={"file": ("clip.wav", wav_bytes, "audio/wav")},
        data={"language": "bm"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "fake transcription"
    assert body["language"] == "bm"
    assert body["model"] == FAKE_ASR
    assert body["duration_s"] > 0


def test_transcribe_audio_url(client, wav_bytes, monkeypatch):
    async def fake_fetch(url):
        assert url == "https://example.com/clip.wav"
        return wav_bytes

    monkeypatch.setattr(transcribe_router, "fetch_audio", fake_fetch)
    resp = client.post(
        "/transcribe",
        json={"audio_url": "https://example.com/clip.wav", "language": "fr"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "fake transcription"
    assert body["language"] == "fr"
    assert body["model"] == FAKE_ASR


def test_transcribe_requires_an_input(client):
    resp = client.post("/transcribe", json={})
    assert resp.status_code == 422


def test_transcribe_rejects_both_inputs(client, wav_bytes):
    resp = client.post(
        "/transcribe",
        files={"file": ("clip.wav", wav_bytes, "audio/wav")},
        data={"audio_url": "https://example.com/clip.wav"},
    )
    assert resp.status_code == 422


def test_transcribe_rejects_undecodable_audio(client):
    resp = client.post(
        "/transcribe",
        files={"file": ("bad.wav", b"not really audio", "audio/wav")},
    )
    assert resp.status_code == 422
