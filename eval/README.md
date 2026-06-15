# Evaluation harness

Scores whatever model is configured (via `MT_MODEL` / `ASR_MODEL`) against the
project targets, so any model change is a measured comparison on the same data.

## Datasets

Provide held-out sets as JSONL (not committed — keep under `eval/data/`):

**Translation** (`eval/data/*.jsonl`):
```json
{"text": "...", "source_lang": "bm", "target_lang": "fr", "reference": "..."}
```

**ASR** (`eval/data/*.jsonl`, `audio_path` relative to the JSONL file):
```json
{"audio_path": "clips/0001.wav", "reference": "...", "language": "bm"}
```

## Running

```bash
uv run python -m eval.translate_eval  eval/data/bm_fr.jsonl
uv run python -m eval.transcribe_eval eval/data/asr.jsonl

# score a specific model without changing config
uv run python -m eval.translate_eval eval/data/bm_fr.jsonl --model <hf-id-or-path>
```

## Targets

- Translation: **BLEU ≥ 35** (primary), ChrF secondary. Exit code 0 when met.
- ASR: **WER ≤ 0.25**. Exit code 0 when met.
