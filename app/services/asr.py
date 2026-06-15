"""ASR service: Whisper-family loader (handles plain models and PEFT/LoRA adapters).

The model id is supplied by config; any Whisper-family model — full checkpoint or
LoRA adapter on a Whisper base — loads here unchanged.
"""

from __future__ import annotations

import numpy as np

TARGET_SR = 16_000

# Whisper's own language codes for the languages we expose. Bambara is not a
# native Whisper language, so we never force a decoder language for it.
_WHISPER_LANG = {"fr": "french", "en": "english"}


class ASRService:
    def __init__(self, model_id: str, device: str = "cpu", hf_token: str | None = None):
        self.model_id = model_id
        self.device = device
        self._hf_token = hf_token
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
            source, token=self._hf_token
        )

        if base is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.model_id, token=self._hf_token)
            model = model.merge_and_unload()

        self._model = model.to(self.device)
        self._model.eval()
        return self

    def transcribe(self, samples: np.ndarray, language: str | None = None) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("ASRService not loaded")

        features = self._processor(
            samples, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(self.device)

        gen_kwargs: dict = {}
        whisper_lang = _WHISPER_LANG.get((language or "").lower())
        if whisper_lang is not None:
            gen_kwargs["forced_decoder_ids"] = self._processor.get_decoder_prompt_ids(
                language=whisper_lang, task="transcribe"
            )

        import torch

        with torch.no_grad():
            generated = self._model.generate(features, **gen_kwargs)
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
