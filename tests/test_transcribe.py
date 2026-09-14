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


class _FakeWhisperClass:
    """Stands in for both WhisperProcessor and WhisperForConditionalGeneration —
    both only need from_pretrained() here, plus .to()/.eval() for the model."""

    def __init__(self):
        self.moved_to = None
        self.eval_called = False

    @classmethod
    def from_pretrained(cls, source, token=None, low_cpu_mem_usage=None):
        instance = cls()
        instance.low_cpu_mem_usage = low_cpu_mem_usage
        return instance

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.eval_called = True
        return self


def _asr_service_with_fake_loading(monkeypatch, quantization):
    """Build an ASRService whose load() never touches the network.

    Real ``transformers`` is a lazily-loaded module (``_LazyModule``); patching
    its attributes directly triggers submodule resolution that ends up
    re-importing the package under a second module object, so the patched
    attribute is invisible from ``app.services.asr``'s own
    ``from transformers import ...``. Swapping the whole module in
    ``sys.modules`` sidesteps that and keeps the fake fully isolated.
    """
    import sys
    import types

    import app.services.asr as asr_module

    monkeypatch.setattr(asr_module.ASRService, "_peft_base", lambda self: None)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.WhisperProcessor = _FakeWhisperClass
    fake_transformers.WhisperForConditionalGeneration = _FakeWhisperClass
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    return asr_module.ASRService("fake/model", device="cpu", quantization=quantization)


def test_asr_load_applies_int8_quantization_on_cpu(monkeypatch):
    import torch

    service = _asr_service_with_fake_loading(monkeypatch, quantization="int8")

    calls = []

    def fake_quantize_dynamic(model, layer_set, dtype):
        calls.append((model, layer_set, dtype))
        return model

    monkeypatch.setattr(
        "torch.ao.quantization.quantize_dynamic", fake_quantize_dynamic
    )

    service.load()

    assert len(calls) == 1
    _, layer_set, dtype = calls[0]
    assert layer_set == {torch.nn.Linear}
    assert dtype == torch.qint8


def test_asr_load_skips_quantization_when_none(monkeypatch):
    service = _asr_service_with_fake_loading(monkeypatch, quantization="none")

    calls = []
    monkeypatch.setattr(
        "torch.ao.quantization.quantize_dynamic",
        lambda *a, **k: calls.append((a, k)),
    )

    service.load()

    assert calls == []


def test_asr_transcribe_works_on_quantized_fake_model():
    """A quantized model is just a different object graph to transcribe() —
    it should behave identically to the unquantized fake-model test above."""
    import numpy as np

    from app.services.asr import ASRService

    class FakeProcessor:
        def __call__(self, samples, sampling_rate=None, return_tensors=None):
            class _F:
                input_features = _Feats()

            return _F()

        def batch_decode(self, generated, skip_special_tokens=True):
            return ["  quantized hello  "]

        def get_decoder_prompt_ids(self, language=None, task=None):
            return [(1, 2)]

    class _Feats:
        def to(self, device):
            return "features"

    class FakeQuantizedModel:
        """Stands in for the object returned by quantize_dynamic."""

        def generate(self, features, **kwargs):
            return [[0]]

    service = ASRService("fake/model", quantization="int8")
    service._processor = FakeProcessor()
    service._model = FakeQuantizedModel()

    out = service.transcribe(np.zeros(16_000, dtype="float32"), language="bm")

    assert out == "quantized hello"


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
