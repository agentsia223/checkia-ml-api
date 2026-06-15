"""Score the configured ASR model on a held-out set.

Dataset: JSONL, one object per line (audio path is relative to the dataset file):
    {"audio_path": "clips/0001.wav", "reference": "...", "language": "bm"}

Usage:
    uv run python -m eval.transcribe_eval eval/data/asr.jsonl
    uv run python -m eval.transcribe_eval eval/data/asr.jsonl --model <hf-id-or-path>

Reports WER (target <= 0.25).
"""

import argparse
import json
import os
import sys

import jiwer

from app.audio import load_audio
from app.config import get_settings
from app.services.asr import ASRService

WER_TARGET = 0.25


def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(path: str, model_id: str, device: str, hf_token: str | None) -> dict:
    rows = load_dataset(path)
    base_dir = os.path.dirname(os.path.abspath(path))
    service = ASRService(model_id, device, hf_token).load()

    hyps, refs = [], []
    for row in rows:
        audio_path = os.path.join(base_dir, row["audio_path"])
        with open(audio_path, "rb") as fh:
            samples, _ = load_audio(fh.read())
        hyps.append(service.transcribe(samples, language=row.get("language")))
        refs.append(row["reference"])

    wer = jiwer.wer(refs, hyps)
    return {"n": len(rows), "model": model_id, "wer": wer}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to JSONL eval set")
    parser.add_argument("--model", default=None, help="Override model id/path")
    args = parser.parse_args()

    settings = get_settings()
    result = run(
        args.dataset,
        args.model or settings.asr_model,
        settings.device,
        settings.hf_token,
    )

    print(f"model: {result['model']}")
    print(f"examples: {result['n']}")
    print(f"WER: {result['wer']:.3f}  (target <= {WER_TARGET})")
    return 0 if result["wer"] <= WER_TARGET else 1


if __name__ == "__main__":
    sys.exit(main())
