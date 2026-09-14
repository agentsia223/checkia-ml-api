"""ASR service: Whisper-family loader (handles plain models and PEFT/LoRA adapters).

The model id is supplied by config; any Whisper-family model — full checkpoint or
LoRA adapter on a Whisper base — loads here unchanged.
"""

from __future__ import annotations

import logging

import numpy as np

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
    ):
        self.model_id = model_id
        self.device = device
        self._hf_token = hf_token
        self.max_new_tokens = max_new_tokens
        self.quantization = quantization
        self.low_cpu_mem_usage = low_cpu_mem_usage
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

    def load(self) -> "ASRService":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        base = self._peft_base()
        source = base or self.model_id

        self._processor = WhisperProcessor.from_pretrained(source, token=self._hf_token)
        model = WhisperForConditionalGeneration.from_pretrained(
            source, token=self._hf_token, low_cpu_mem_usage=self.low_cpu_mem_usage
        )

        if base is not None:
            from peft import PeftModel

            peft_model = PeftModel.from_pretrained(model, self.model_id, token=self._hf_token)
            model = peft_model.merge_and_unload()
            del peft_model

        if self.quantization == "int8":
            if self.device == "cpu":
                import torch

                try:
                    from torch.ao.quantization import quantize_dynamic
                except ImportError:
                    from torch.quantization import quantize_dynamic

                # inplace=True: the default (False) deep-copies the entire
                # fp32 model before converting, roughly tripling peak memory
                # during load (fp32 original + fp32 copy + int8 result).
                model = quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
                )
                logger.info("ASR model quantized: dynamic int8 (torch.nn.Linear)")
            else:
                logger.warning(
                    "asr_quantization=int8 requested but device=%s; dynamic int8 "
                    "quantization is CPU-only, skipping",
                    self.device,
                )
        else:
            logger.info("ASR model quantization: none")

        model = model.to(self.device)
        model.eval()

        rss_before = _get_rss_mb()
        if rss_before is not None:
            logger.info("ASR loaded: RSS ≈ %.0f MB (before cleanup)", rss_before)

        _release_memory_to_os()

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
