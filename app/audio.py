"""Audio decoding helpers: bytes/URL -> 16 kHz mono float32 samples."""

from __future__ import annotations

import io
import tempfile

import numpy as np

TARGET_SR = 16_000


def load_audio(data: bytes) -> tuple[np.ndarray, float]:
    """Decode audio bytes to mono 16 kHz float32 and return (samples, duration_s).

    Tries ``soundfile`` first (wav/flac/ogg); falls back to ``librosa`` (which
    can use audioread/ffmpeg for mp3/m4a). Duration is measured at the original
    sample rate before resampling.
    """
    import soundfile as sf

    try:
        samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        import librosa

        with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
            tmp.write(data)
            tmp.flush()
            samples, sr = librosa.load(tmp.name, sr=None, mono=True)

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:  # stereo -> mono
        samples = samples.mean(axis=1)

    duration_s = float(len(samples) / sr) if sr else 0.0

    if sr != TARGET_SR:
        import librosa

        samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SR)

    return samples, duration_s


async def fetch_audio(url: str) -> bytes:
    """Download audio bytes from a URL. Raises on any HTTP/network error."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content
