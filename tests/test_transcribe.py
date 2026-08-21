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


def test_asr_service_caps_decoding_length():
    """The generation cap is what bounds latency: Whisper otherwise decodes up to
    max_length (448) sequential steps regardless of how short the audio is."""
    import numpy as np

    from app.services.asr import ASRService

    captured = {}

    class FakeProcessor:
        def __call__(self, samples, sampling_rate=None, return_tensors=None):
            class _F:
                input_features = _Feats()

            return _F()

        def batch_decode(self, generated, skip_special_tokens=True):
            return ["  hello  "]

        def get_decoder_prompt_ids(self, language=None, task=None):
            return [(1, 2)]

    class _Feats:
        def to(self, device):
            return "features"

    class FakeModel:
        def generate(self, features, **kwargs):
            captured.update(kwargs)
            return [[0]]

    service = ASRService("fake/model", max_new_tokens=123)
    service._processor = FakeProcessor()
    service._model = FakeModel()

    out = service.transcribe(np.zeros(16_000, dtype="float32"), language="bm")

    assert out == "hello"
    assert captured["max_new_tokens"] == 123
    # Bambara is not a Whisper language, so no decoder language is forced.
    assert "forced_decoder_ids" not in captured


def test_transcribe_does_not_block_the_event_loop(wav_bytes, monkeypatch):
    """/transcribe is an async endpoint doing blocking CPU work. If that work runs
    inline, it stalls the event loop and every other request (/health, /translate)
    waits behind it. Here a slow transcription runs concurrently with /health:
    /health must still answer promptly."""
    import asyncio
    import time

    import httpx

    from app.main import _noop_lifespan, create_app
    from tests.conftest import FAKE_ASR, FAKE_MT, FakeASRService, FakeTranslationService
    from app.config import Settings

    BLOCK_SECONDS = 1.0

    app = create_app(lifespan=_noop_lifespan)
    app.state.settings = Settings(
        _env_file=None, mt_model=FAKE_MT, asr_model=FAKE_ASR, device="cpu"
    )
    app.state.translation = FakeTranslationService()
    app.state.asr = FakeASRService()

    def blocking_transcribe(samples, language=None):
        time.sleep(BLOCK_SECONDS)  # stands in for CPU-bound decoding
        return "fake transcription"

    monkeypatch.setattr(app.state.asr, "transcribe", blocking_transcribe)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            started = time.perf_counter()
            # Both in flight at once. If the transcription blocks the loop, the
            # /health task cannot even start until it finishes.
            slow = asyncio.create_task(
                ac.post(
                    "/transcribe",
                    files={"file": ("clip.wav", wav_bytes, "audio/wav")},
                    data={"language": "bm"},
                )
            )
            health_task = asyncio.create_task(ac.get("/health"))
            health = await health_task
            health_latency = time.perf_counter() - started
            return health, health_latency, await slow

    health, health_latency, transcribed = asyncio.run(scenario())

    assert transcribed.status_code == 200
    assert transcribed.json()["text"] == "fake transcription"
    assert health.status_code == 200
    # If the loop were blocked, /health would wait out the full transcription.
    assert health_latency < BLOCK_SECONDS / 2, (
        f"/health took {health_latency:.2f}s while a transcription ran — "
        "the event loop is blocked"
    )
