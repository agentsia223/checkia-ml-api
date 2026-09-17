"""Build and cache quantized ASR artifacts on disk.

Loading the fp32 Whisper checkpoint, merging PEFT weights, and quantizing to
int8 all happen in this module's ``load_and_quantize``/``build_artifact``
functions. The point of running that in a *separate process* (see
``ASRService.load`` in ``asr.py``) is that the fp32 pages never have to be
resident in the long-lived serving process — the subprocess allocates them,
quantizes, writes the result to disk, and exits, handing the OS its memory
back for free.

Run directly as a CLI to (re)build the cache:

    python -m app.services.asr_cache --model-id ... --device cpu \
        --quantization int8 --cache-dir /var/cache/asr
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time

logger = logging.getLogger(__name__)


def cache_key(model_id: str, quantization: str) -> str:
    """Short, stable key for a (model, quantization, library-versions) combo.

    Pickled quantized modules are sensitive to the exact torch/transformers
    versions that produced them, so both are folded into the key: a version
    bump invalidates the cache instead of risking a bad unpickle.
    """
    import torch
    import transformers

    raw = f"{model_id}|{quantization}|torch={torch.__version__}|transformers={transformers.__version__}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def artifact_paths(cache_dir: str, key: str) -> tuple[str, str]:
    """Return (model_path, processor_dir) for the given cache key."""
    base = os.path.join(cache_dir, key)
    return os.path.join(base, "model.pt"), os.path.join(base, "processor")


def load_and_quantize(
    model_id: str,
    device: str,
    hf_token: str | None,
    quantization: str,
    low_cpu_mem_usage: bool = True,
):
    """Load a Whisper(-family) checkpoint, merge any PEFT adapter, quantize.

    This is the loading logic previously inlined in ``ASRService.load()``;
    it is intentionally free of any ``ASRService`` state so it can run either
    in-process (legacy path) or in the cache-builder subprocess.

    Returns (processor, model).
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    base = None
    try:
        from peft import PeftConfig

        cfg = PeftConfig.from_pretrained(model_id, token=hf_token)
        base = cfg.base_model_name_or_path
    except Exception:
        base = None

    source = base or model_id

    processor = WhisperProcessor.from_pretrained(source, token=hf_token)
    model = WhisperForConditionalGeneration.from_pretrained(
        source, token=hf_token, low_cpu_mem_usage=low_cpu_mem_usage
    )

    if base is not None:
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(model, model_id, token=hf_token)
        model = peft_model.merge_and_unload()
        del peft_model

    if quantization == "int8":
        if device == "cpu":
            import torch

            try:
                from torch.ao.quantization import quantize_dynamic
            except ImportError:
                from torch.quantization import quantize_dynamic

            # inplace=True: the default (False) deep-copies the entire fp32
            # model before converting, roughly tripling peak memory during
            # load (fp32 original + fp32 copy + int8 result).
            model = quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
            logger.info("ASR model quantized: dynamic int8 (torch.nn.Linear)")
        else:
            logger.warning(
                "asr_quantization=int8 requested but device=%s; dynamic int8 "
                "quantization is CPU-only, skipping",
                device,
            )
    else:
        logger.info("ASR model quantization: none")

    model = model.to(device)
    model.eval()

    return processor, model


def build_artifact(
    model_id: str,
    device: str,
    hf_token: str | None,
    quantization: str,
    cache_dir: str,
    low_cpu_mem_usage: bool = True,
) -> str:
    """Build the quantized artifact and atomically write it to the cache.

    Returns the cache key used.
    """
    import torch

    key = cache_key(model_id, quantization)
    model_path, processor_dir = artifact_paths(cache_dir, key)
    base_dir = os.path.dirname(model_path)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(processor_dir, exist_ok=True)

    started = time.monotonic()
    processor, model = load_and_quantize(
        model_id, device, hf_token, quantization, low_cpu_mem_usage
    )

    processor.save_pretrained(processor_dir)

    tmp_path = model_path + f".tmp-{os.getpid()}"
    torch.save(model, tmp_path)
    size_mb = 0.0
    try:
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    except OSError:
        pass
    os.replace(tmp_path, model_path)

    elapsed = time.monotonic() - started
    logger.info(
        "ASR cache artifact built: key=%s size=%.0fMB elapsed=%.1fs path=%s",
        key, size_mb, elapsed, model_path,
    )
    return key


def _main() -> None:
    from .. import runtime

    runtime.configure_thread_env()

    parser = argparse.ArgumentParser(description="Build a quantized ASR cache artifact")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quantization", default="int8")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--low-cpu-mem-usage",
        dest="low_cpu_mem_usage",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-low-cpu-mem-usage", dest="low_cpu_mem_usage", action="store_false"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    hf_token = args.hf_token
    if not hf_token:
        from ..config import get_settings

        hf_token = get_settings().hf_token

    build_artifact(
        args.model_id,
        args.device,
        hf_token,
        args.quantization,
        args.cache_dir,
        args.low_cpu_mem_usage,
    )


if __name__ == "__main__":
    _main()
