"""Tests for the on-disk quantized-ASR-artifact cache (app/services/asr_cache.py)
and the cache-aware ASRService.load() path in app/services/asr.py.

This Mac has no quantization engine (NoQEngine), so quantize_dynamic is never
actually invoked here — it's faked, same as the existing asr tests do.
"""

import subprocess
import sys
import types

import pytest


def _fake_transformers(monkeypatch):
    """Install a fake `transformers` module, same trick test_transcribe.py uses."""
    import app.services.asr as asr_module
    import app.services.asr_cache as asr_cache_module

    class _FakeWhisperClass:
        def __init__(self):
            self.moved_to = None

        @classmethod
        def from_pretrained(cls, source, token=None, low_cpu_mem_usage=None):
            return cls()

        def to(self, device):
            self.moved_to = device
            return self

        def eval(self):
            return self

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.WhisperProcessor = _FakeWhisperClass
    fake_transformers.WhisperForConditionalGeneration = _FakeWhisperClass
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        asr_cache_module, "load_and_quantize",
        asr_cache_module.load_and_quantize,  # keep real function; it imports the fake module
    )
    return _FakeWhisperClass


def test_cache_key_differs_by_quantization():
    from app.services.asr_cache import cache_key

    k1 = cache_key("model-a", "int8")
    k2 = cache_key("model-a", "none")
    assert k1 != k2


def test_cache_key_stable_for_same_inputs():
    from app.services.asr_cache import cache_key

    assert cache_key("model-a", "int8") == cache_key("model-a", "int8")


def test_artifact_paths(tmp_path):
    from app.services.asr_cache import artifact_paths

    model_path, processor_dir = artifact_paths(str(tmp_path), "abc123")
    assert model_path == str(tmp_path / "abc123" / "model.pt")
    assert processor_dir == str(tmp_path / "abc123" / "processor")


def test_asr_load_legacy_path_when_cache_disabled(monkeypatch):
    """cache_dir falsy -> legacy in-process load, no subprocess, no torch.load."""
    import app.services.asr as asr_module

    _fake_transformers(monkeypatch)
    monkeypatch.setattr(asr_module.ASRService, "_peft_base", lambda self: None)
    monkeypatch.setattr(
        "torch.ao.quantization.quantize_dynamic", lambda model, *a, **k: model
    )

    calls = {"subprocess": 0, "torch_load": 0}
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: calls.__setitem__("subprocess", calls["subprocess"] + 1)
    )
    import torch
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: calls.__setitem__("torch_load", calls["torch_load"] + 1)
    )

    service = asr_module.ASRService("fake/model", device="cpu", quantization="int8", cache_dir=None)
    service.load()

    assert calls["subprocess"] == 0
    assert calls["torch_load"] == 0


def test_asr_load_cache_hit(monkeypatch, tmp_path):
    """model.pt + processor already exist -> torch.load used, no subprocess, no from_pretrained rebuild."""
    import app.services.asr as asr_module
    from app.services.asr_cache import artifact_paths, cache_key

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "0.0.0-test"
    fake_transformers.WhisperProcessor = FakeProcessor
    fake_transformers.WhisperForConditionalGeneration = FakeProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    key = cache_key("fake/model", "int8")
    model_path, processor_dir = artifact_paths(str(tmp_path), key)
    import os
    os.makedirs(processor_dir, exist_ok=True)
    with open(model_path, "wb") as f:
        f.write(b"fake-pickle")

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    calls = {"subprocess": 0, "torch_load": []}
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: calls.__setitem__("subprocess", calls["subprocess"] + 1)
    )
    import torch
    monkeypatch.setattr(
        torch, "load",
        lambda path, map_location=None, weights_only=None: (
            calls["torch_load"].append(path), FakeModel()
        )[1],
    )

    service = asr_module.ASRService(
        "fake/model", device="cpu", quantization="int8", cache_dir=str(tmp_path)
    )
    service.load()

    assert calls["subprocess"] == 0
    assert calls["torch_load"] == [model_path]
    assert service._model is not None


def test_asr_load_cache_miss_builds_via_subprocess(monkeypatch, tmp_path):
    """No cached artifact -> subprocess.run builds it, then torch.load reads it."""
    import app.services.asr as asr_module
    from app.services.asr_cache import artifact_paths, cache_key

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "0.0.0-test"
    fake_transformers.WhisperProcessor = FakeProcessor
    fake_transformers.WhisperForConditionalGeneration = FakeProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    key = cache_key("fake/model", "int8")
    model_path, processor_dir = artifact_paths(str(tmp_path), key)

    subprocess_calls = []

    def fake_run(args, check=None, env=None):
        subprocess_calls.append(args)
        # Simulate the builder creating the artifact.
        import os
        os.makedirs(processor_dir, exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(b"fake-pickle")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    import torch
    monkeypatch.setattr(torch, "load", lambda path, map_location=None, weights_only=None: FakeModel())

    service = asr_module.ASRService(
        "fake/model", device="cpu", quantization="int8", cache_dir=str(tmp_path)
    )
    service.load()

    assert len(subprocess_calls) == 1
    args = subprocess_calls[0]
    assert args[0] == sys.executable
    assert args[1:3] == ["-m", "app.services.asr_cache"]
    assert "--model-id" in args and "fake/model" in args
    assert "--device" in args and "cpu" in args
    assert "--quantization" in args and "int8" in args
    assert "--cache-dir" in args and str(tmp_path) in args


def test_asr_load_falls_back_to_legacy_on_subprocess_failure(monkeypatch, tmp_path, caplog):
    """Builder subprocess fails -> service still comes up via the legacy path."""
    import app.services.asr as asr_module

    _fake_transformers(monkeypatch)
    monkeypatch.setattr(asr_module.ASRService, "_peft_base", lambda self: None)
    monkeypatch.setattr(
        "torch.ao.quantization.quantize_dynamic", lambda model, *a, **k: model
    )

    def fake_run(args, check=None, env=None):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "run", fake_run)

    caplog.set_level("ERROR")

    service = asr_module.ASRService(
        "fake/model", device="cpu", quantization="int8", cache_dir=str(tmp_path)
    )
    service.load()

    assert service._model is not None
    assert any("falling back" in r.message.lower() for r in caplog.records)


def test_footprint_classifies_quantized_linear_by_module_path():
    from app.services.asr import _model_footprint

    import torch

    class _QuantizedLinear:
        __module__ = "torch.ao.nn.quantized.dynamic.modules.linear"
        __qualname__ = "Linear"
        __name__ = "Linear"

        def weight(self):
            return torch.zeros(1)

    _QuantizedLinear.__name__ = "Linear"

    class _FloatLinear:
        __module__ = "torch.nn.modules.linear"

    _FloatLinear.__name__ = "Linear"

    class FakeModel(torch.nn.Module):
        def __init__(self, submodules):
            super().__init__()
            self._submodules = submodules

        def parameters(self):
            return iter(())

        def buffers(self):
            return iter(())

        def modules(self):
            return iter(self._submodules)

    model = FakeModel([_QuantizedLinear(), _FloatLinear()])
    _, n_q, n_f = _model_footprint(model)

    assert n_q == 1
    assert n_f == 1


def test_build_artifact_writes_processor_and_model(monkeypatch, tmp_path):
    import app.services.asr_cache as asr_cache_module

    class FakeProcessor:
        def save_pretrained(self, path):
            import os
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "preprocessor_config.json"), "w") as f:
                f.write("{}")

    class FakeModel:
        pass

    def fake_load_and_quantize(model_id, device, hf_token, quantization, low_cpu_mem_usage=True):
        return FakeProcessor(), FakeModel()

    monkeypatch.setattr(asr_cache_module, "load_and_quantize", fake_load_and_quantize)

    save_calls = []

    def fake_torch_save(obj, path):
        save_calls.append(path)
        with open(path, "wb") as f:
            f.write(b"pickled")

    import torch
    monkeypatch.setattr(torch, "save", fake_torch_save)

    key = asr_cache_module.build_artifact(
        "fake/model", "cpu", None, "int8", str(tmp_path)
    )

    model_path, processor_dir = asr_cache_module.artifact_paths(str(tmp_path), key)
    assert len(save_calls) == 1
    assert save_calls[0] != model_path  # written to a tmp path first
    assert __import__("os").path.exists(model_path)
    assert __import__("os").path.exists(
        __import__("os").path.join(processor_dir, "preprocessor_config.json")
    )
