"""ASR service: Whisper-family loader (handles plain models and PEFT/LoRA adapters).

The model id is supplied by config; any Whisper-family model — full checkpoint or
LoRA adapter on a Whisper base — loads here unchanged.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from time import monotonic as _monotonic

import numpy as np

from .asr_cache import artifact_paths, cache_key, load_and_quantize

logger = logging.getLogger(__name__)

TARGET_SR = 16_000

# Whisper's own language codes for the languages we expose. Bambara is not a
# native Whisper language, so we never force a decoder language for it.
_WHISPER_LANG = {"fr": "french", "en": "english"}


def _get_rss_mb() -> float | None:
    """Read current RSS (in MB) from /proc/self/status, if available."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024
    except (OSError, ValueError, IndexError):
        return None
    return None



def _model_footprint(model) -> tuple[float, int, int]:
    """Return (tensor_mb, n_quantized_linear, n_float_linear) for a torch module.

    tensor_mb is the real size of parameters + buffers, independent of RSS, so the
    two numbers together separate "quantization did not take" from "float weights
    still referenced" from "allocator holding freed pages".
    """
    try:
        import torch

        if not isinstance(model, torch.nn.Module):
            return 0.0, 0, 0
        total = sum(t.numel() * t.element_size() for t in model.parameters())
        total += sum(t.numel() * t.element_size() for t in model.buffers())
        # Dynamic-quantized Linear layers store packed int8 weights, not parameters.
        # torch's quantized Linear is ALSO named plain "Linear" (it lives in
        # torch.ao.nn.quantized.dynamic.modules.linear), so classify by module
        # path, not class name — name-only classification undercounts quantized
        # layers as float ones.
        n_q = n_f = 0
        for m in model.modules():
            module_path = type(m).__module__
            name = type(m).__name__
            if "quantized" in module_path:
                if name == "Linear":
                    n_q += 1
                    try:
                        w = m.weight()  # packed weight accessor
                        total += w.numel() * w.element_size()
                    except Exception:
                        pass
            elif name == "Linear":
                n_f += 1
        return total / (1024 * 1024), n_q, n_f
    except Exception:
        return 0.0, 0, 0

def _release_memory_to_os() -> None:
    """Force a GC pass and return freed heap pages to the OS (glibc only).

    quantize_dynamic and the PEFT merge leave large freed allocations sitting
    in glibc's malloc arenas that are never handed back to the OS on their
    own, inflating steady-state RSS well past the size of the live model.
    malloc_trim(0) reclaims them. No-op (and safe) on non-glibc platforms
    (macOS dev, musl).
    """
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # non-glibc platform (macOS dev, musl)


class ASRService:
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        hf_token: str | None = None,
        max_new_tokens: int = 200,
        quantization: str = "int8",
        low_cpu_mem_usage: bool = True,
        cache_dir: str | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self._hf_token = hf_token
        self.max_new_tokens = max_new_tokens
        self.quantization = quantization
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.cache_dir = cache_dir
        self._processor = None
        self._model = None

    @property
    def model_name(self) -> str:
        return self.model_id

    def _peft_base(self) -> str | None:
        """Return the base model id if model_id is a PEFT adapter, else None."""
        try:
            from peft import PeftConfig

            cfg = PeftConfig.from_pretrained(self.model_id, token=self._hf_token)
            return cfg.base_model_name_or_path
        except Exception:
            return None

    def _load_legacy(self) -> tuple[object, object]:
        """Load, merge, and quantize the model in this process (fp32 briefly resident)."""
        return load_and_quantize(
            self.model_id,
            self.device,
            self._hf_token,
            self.quantization,
            self.low_cpu_mem_usage,
        )

    def _build_cache_args(self, cache_dir: str) -> list[str]:
        args = [
            sys.executable,
            "-m",
            "app.services.asr_cache",
            "--model-id",
            self.model_id,
            "--device",
            self.device,
            "--quantization",
            self.quantization,
            "--cache-dir",
            cache_dir,
        ]
        if not self.low_cpu_mem_usage:
            args.append("--no-low-cpu-mem-usage")
        # The token is deliberately NOT passed on the command line (it would show in
        # the process list); the builder reads it from the inherited environment.
        return args

    def _load_from_cache(self, cache_dir: str) -> tuple[object, object]:
        import torch
        from transformers import WhisperProcessor

        key = cache_key(self.model_id, self.quantization)
        model_path, processor_dir = artifact_paths(cache_dir, key)

        if not os.path.exists(model_path):
            args = self._build_cache_args(cache_dir)
            started = _monotonic()
            subprocess.run(args, check=True, env=os.environ.copy())
            logger.info(
                "ASR cache artifact built by subprocess in %.1fs (key=%s)",
                _monotonic() - started, key,
            )

        model = torch.load(model_path, map_location=self.device, weights_only=False)
        processor = WhisperProcessor.from_pretrained(processor_dir)
        model.eval()
        logger.info("ASR loaded from cache: key=%s", key)
        return processor, model

    def load(self) -> "ASRService":
        source_label = "legacy"
        if self.cache_dir:
            try:
                self._processor, model = self._load_from_cache(self.cache_dir)
                source_label = "cache"
            except Exception as exc:
                logger.error(
                    "ASR cache load failed (%s: %s); falling back to legacy in-process load",
                    type(exc).__name__, exc,
                )
                self._processor, model = self._load_legacy()
                source_label = "legacy"
        else:
            self._processor, model = self._load_legacy()

        model = model.to(self.device)
        model.eval()

        rss_before = _get_rss_mb()
        if rss_before is not None:
            logger.info("ASR loaded: RSS ≈ %.0f MB (before cleanup)", rss_before)

        _release_memory_to_os()
        tensor_mb, n_q, n_f = _model_footprint(model)
        logger.info(
            "ASR footprint: tensors ≈ %.0f MB, quantized Linear=%d, float Linear=%d, "
            "RSS ≈ %s MB, source=%s",
            tensor_mb, n_q, n_f, _get_rss_mb(), source_label,
        )

        rss_after = _get_rss_mb()
        if rss_after is not None:
            logger.info("ASR loaded: RSS ≈ %.0f MB", rss_after)

        self._model = model
        return self

    def transcribe(self, samples: np.ndarray, language: str | None = None) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("ASRService not loaded")

        features = self._processor(
            samples, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(self.device)

        # Bound the decode. Without this Whisper may generate up to its
        # max_length (448) even for a few seconds of audio, which dominates
        # latency because decoding is sequential.
        gen_kwargs: dict = {"max_new_tokens": self.max_new_tokens}
        whisper_lang = _WHISPER_LANG.get((language or "").lower())
        if whisper_lang is not None:
            gen_kwargs["forced_decoder_ids"] = self._processor.get_decoder_prompt_ids(
                language=whisper_lang, task="transcribe"
            )

        import torch

        with torch.no_grad():
            generated = self._model.generate(features, **gen_kwargs)
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
