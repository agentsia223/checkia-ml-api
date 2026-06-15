"""Score the configured translation model on a held-out set.

Dataset: JSONL, one object per line:
    {"text": "...", "source_lang": "bm", "target_lang": "fr", "reference": "..."}

Usage:
    uv run python -m eval.translate_eval eval/data/bm_fr.jsonl
    uv run python -m eval.translate_eval eval/data/bm_fr.jsonl --model <hf-id-or-path>

Reports BLEU (primary, target >= 35) and ChrF (secondary).
"""

import argparse
import json
import sys

import sacrebleu

from app.config import get_settings
from app.services.translation import GoogleTranslationService

BLEU_TARGET = 35.0


def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(path: str, model_id: str, api_key: str | None, timeout: float) -> dict:
    rows = load_dataset(path)
    service = GoogleTranslationService(model_id, api_key, timeout).load()

    hyps = [service.translate(r["text"], r["source_lang"], r["target_lang"]) for r in rows]
    refs = [[r["reference"] for r in rows]]

    bleu = sacrebleu.corpus_bleu(hyps, refs).score
    chrf = sacrebleu.corpus_chrf(hyps, refs).score
    return {"n": len(rows), "model": model_id, "bleu": bleu, "chrf": chrf}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to JSONL eval set")
    parser.add_argument("--model", default=None, help="Override model id/path")
    args = parser.parse_args()

    settings = get_settings()
    result = run(
        args.dataset,
        args.model or settings.mt_model,
        settings.google_translate_api_key,
        settings.translate_timeout,
    )

    print(f"model: {result['model']}")
    print(f"examples: {result['n']}")
    print(f"BLEU: {result['bleu']:.2f}  (target >= {BLEU_TARGET})")
    print(f"ChrF: {result['chrf']:.2f}")
    return 0 if result["bleu"] >= BLEU_TARGET else 1


if __name__ == "__main__":
    sys.exit(main())
