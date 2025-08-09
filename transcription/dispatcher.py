"""Google Cloud Speech–to–Text integration with speaker diarisation (GCS + LRO)."""

from __future__ import annotations

import os
from typing import Dict, Any

from .whisper_transcriber import transcribe_whisper
from .google_transcriber import transcribe_google


def transcribe(
    audio_path: str,
    use_diarization: bool = False,
    *,
    # keep these tweakable so you don't hardcode in the GCS module
    language_code: str = "en-US",
    sample_rate_hz: int = 16000,   # must match your preprocess_audio output
    min_speakers: int = 2,
    max_speakers: int = 6,
) -> Dict[str, Any]:
    """
    Transcribe an audio recording. When `use_diarization` is True, use Google STT
    via LongRunningRecognize with a GCS URI; otherwise use Whisper.
    """
    if use_diarization:
        gcs_bucket = "my-transcriber-temp-bucket"
        if not gcs_bucket:
            # Optional: soft-fail to Whisper instead of raising
            raise RuntimeError("GCS_BUCKET not set. Set env var or pass a bucket to use diarization.")
        return transcribe_google(
            audio_path,
            gcs_bucket=gcs_bucket,
            language_code=language_code,
            sample_rate_hz=sample_rate_hz,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            delete_after=True,
        )

    return transcribe_whisper(audio_path)
