"""Audio decoding (FFmpeg pipe, no temp files) and a basic usability gate."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from avdd.common.errors import AudioError

SAMPLE_RATE = 16_000
_FRAME = 400  # 25 ms at 16 kHz


def decode_audio(
    path: str | Path,
    max_seconds: float = 60.0,
    timeout_s: float = 60.0,
    ffmpeg_bin: str = "ffmpeg",
) -> NDArray[np.float32]:
    """Decode the first audio stream to 16 kHz mono float32 in [-1, 1].

    Only the first ``max_seconds`` are decoded. Raises ``AudioError`` with a stable
    code: file_not_found, tool_missing, timeout, decode_failed, no_audio_samples,
    invalid_samples.
    """
    p = Path(path)
    if not p.is_file():
        raise AudioError("file_not_found", "input file not found")
    if shutil.which(ffmpeg_bin) is None:
        raise AudioError("tool_missing", "ffmpeg not found on PATH")
    cmd = [
        ffmpeg_bin, "-nostdin", "-v", "error", "-i", str(p.resolve()), "-map", "0:a:0",
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-t", str(max_seconds), "-f", "f32le", "pipe:1",
    ]  # fmt: skip
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AudioError("timeout", f"audio decoding exceeded {timeout_s:.0f}s") from exc
    if proc.returncode != 0:
        raise AudioError("decode_failed", "audio stream could not be decoded")
    if len(proc.stdout) < 4:
        raise AudioError("no_audio_samples", "decoded audio is empty")
    wave: NDArray[np.float32] = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if not bool(np.isfinite(wave).all()):
        raise AudioError("invalid_samples", "decoded audio contains non-finite samples")
    return wave


def assess_audio(
    wave: NDArray[np.float32],
    min_duration_s: float = 1.0,
    min_rms_dbfs: float = -55.0,
    active_dbfs: float = -45.0,
    min_active_ratio: float = 0.15,
) -> tuple[list[str], dict[str, Any]]:
    """Energy-based usability gate. Returns (reasons_unreliable, metrics).

    This only detects short, near-silent, or very low-energy audio. It cannot tell
    speech from music or noise, yet the detector is meaningful only for speech.
    """
    n = int(wave.shape[0])
    duration = n / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(np.square(wave, dtype=np.float64)))) if n else 0.0
    rms_db = 20.0 * float(np.log10(max(rms, 1e-9)))
    n_frames = n // _FRAME
    if n_frames:
        frames = wave[: n_frames * _FRAME].reshape(n_frames, _FRAME).astype(np.float64)
        frame_db = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(frames**2, axis=1)), 1e-9))
        active_ratio = float(np.mean(frame_db > active_dbfs))
    else:
        active_ratio = 0.0
    reasons: list[str] = []
    if duration < min_duration_s:
        reasons.append(f"audio shorter than {min_duration_s:g} s")
    if rms_db < min_rms_dbfs:
        reasons.append("near-silent audio")
    elif active_ratio < min_active_ratio:
        reasons.append(f"less than {min_active_ratio:.0%} of audio frames above the silence floor")
    metrics = {"duration_s": round(duration, 3), "rms_dbfs": round(rms_db, 1),
               "active_frame_ratio": round(active_ratio, 3)}  # fmt: skip
    return reasons, metrics


def audio_input_diagnostics(
    wave: NDArray[np.float32], max_seconds: float, clip_level: float = 0.999
) -> dict[str, Any]:
    """Facts about the decoded waveform that can explain an odd detector output.

    ``speech_band_ratio`` is a crude spectral heuristic (share of energy between 300 and 3400 Hz).
    It is NOT a speech detector and is shown only as context.
    """
    n = int(wave.shape[0])
    duration = n / SAMPLE_RATE
    peak = float(np.max(np.abs(wave))) if n else 0.0
    clipping = float(np.mean(np.abs(wave) >= clip_level)) if n else 0.0
    spectrum = np.abs(np.fft.rfft(wave.astype(np.float64))) ** 2 if n else np.zeros(1)
    freqs = np.fft.rfftfreq(max(n, 1), d=1.0 / SAMPLE_RATE)
    total = float(spectrum.sum())
    band = float(spectrum[(freqs >= 300.0) & (freqs <= 3400.0)].sum())
    return {
        "target_sample_rate": SAMPLE_RATE,
        "channels_after_downmix": 1,
        "normalization": "none (waveform passed to the model as decoded, range [-1, 1])",
        "decoded_s": round(duration, 3),
        "truncated_to_limit": duration >= max_seconds - 0.01,
        "peak_abs": round(peak, 4),
        "clipping_ratio": round(clipping, 5),
        "speech_band_ratio": round(band / total, 3) if total > 0 else None,
    }
