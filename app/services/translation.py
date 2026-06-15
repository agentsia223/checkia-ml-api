"""Translation service: sequence-to-sequence (NLLB / M2M-100 family) loader.

The model id is supplied by config; any same-family model loads here unchanged.
"""

from __future__ import annotations

from ..lang import to_nllb

MAX_NEW_TOKENS = 512


class TranslationService:
    def __init__(self, model_id: str, device: str = "cpu", hf_token: str | None = None):
        self.model_id = model_id
        self.device = device
        self._hf_token = hf_token
        self._tokenizer = None
        self._model = None

    @property
    def model_name(self) -> str:
        return self.model_id

    def load(self) -> "TranslationService":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=self._hf_token
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, token=self._hf_token
        ).to(self.device)
        self._model.eval()
        return self

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("TranslationService not loaded")

        src = to_nllb(source_lang)
        tgt = to_nllb(target_lang)

        self._tokenizer.src_lang = src
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        forced_bos = self._tokenizer.convert_tokens_to_ids(tgt)

        import torch

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
