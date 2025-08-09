"""Google Cloud Speech–to–Text integration with speaker diarisation (GCS + LRO).

This module uploads audio to a Google Cloud Storage bucket and uses the
LongRunningRecognize API to handle files longer than 1 minute, with
speaker diarisation enabled. It groups words by speaker into timestamped
segments.

Prerequisites:
- google-cloud-speech and google-cloud-storage installed.
- GOOGLE_APPLICATION_CREDENTIALS env var set to your service account JSON key.
- GCS_BUCKET env var set to the name of your Cloud Storage bucket.
"""

from __future__ import annotations

import os
import uuid
from typing import Dict, Any, List, Optional

try:
    from google.cloud import speech_v1 as speech  # type: ignore
    from google.cloud import storage  # type: ignore
except ModuleNotFoundError:
    speech = None  # type: ignore
    storage = None  # type: ignore

# Imported for completeness so tools know it's an expected config value.
from config import GOOGLE_APPLICATION_CREDENTIALS  # type: ignore  # noqa: F401


# --------------------------
# Internal helper functions
# --------------------------

def _upload_to_gcs(local_path: str, bucket_name: str, prefix: str = "uploads/") -> str:
    """Upload a local file to GCS and return its gs:// URI."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"{prefix}{uuid.uuid4().hex}_{os.path.basename(local_path)}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"


def _group_words_into_segments(words: List[speech.WordInfo]) -> List[Dict[str, Any]]:
    """Group consecutive words with the same speaker_tag into segments."""
    segments: List[Dict[str, Any]] = []
    if not words:
        return segments

    def ts_to_sec(ts) -> float:
        return ts.total_seconds() if ts is not None else 0.0

    cur_speaker: Optional[int] = None
    cur_words: List[str] = []
    seg_start: Optional[float] = None
    seg_end: Optional[float] = None

    for w in words:
        spk = w.speaker_tag
        start = ts_to_sec(w.start_time)
        end = ts_to_sec(w.end_time)

        if cur_speaker is None:
            cur_speaker = spk
            cur_words = [w.word]
            seg_start = start
            seg_end = end
            continue

        if spk == cur_speaker:
            cur_words.append(w.word)
            seg_end = end
        else:
            segments.append({
                "speaker": f"SPEAKER_{cur_speaker}",
                "start": seg_start or 0.0,
                "end": seg_end or seg_start or 0.0,
                "text": " ".join(cur_words).strip(),
            })
            cur_speaker = spk
            cur_words = [w.word]
            seg_start = start
            seg_end = end

    if cur_words:
        segments.append({
            "speaker": f"SPEAKER_{cur_speaker}",
            "start": seg_start or 0.0,
            "end": seg_end or seg_start or 0.0,
            "text": " ".join(cur_words).strip(),
        })

    return segments


# --------------------------
# Public API
# --------------------------

def transcribe_google(
    audio_path: str,
    *,
    gcs_bucket: Optional[str] = None,
    language_code: str = "en-US",
    sample_rate_hz: int = 16000,  # match preprocess_audio output
    min_speakers: int = 2,
    max_speakers: int = 6,
    delete_after: bool = True,
) -> Dict[str, Any]:
    """
    Transcribe audio using Google Cloud Speech-to-Text with diarisation
    via LongRunningRecognize.

    Args:
        audio_path: Path to the audio file (mono LINEAR16 WAV recommended).
        gcs_bucket: GCS bucket to upload to (or set GCS_BUCKET env var).
        language_code: Language code for transcription.
        sample_rate_hz: Audio sample rate in Hz.
        min_speakers: Minimum expected speaker count.
        max_speakers: Maximum expected speaker count.
        delete_after: Delete uploaded GCS object after processing.

    Returns:
        dict with keys:
          - "text": full transcript
          - "segments": list of speaker-tagged segments with timestamps
          - "full_response": raw API response
          - "gcs_uri": GCS URI used for recognition
    """
    if speech is None or storage is None:
        raise RuntimeError(
            "google-cloud-speech and google-cloud-storage are required. "
            "Install them to use Google transcription."
        )

    bucket_name = gcs_bucket or os.environ.get("GCS_BUCKET")
    if not bucket_name:
        raise RuntimeError("No GCS bucket specified. Set gcs_bucket param or GCS_BUCKET env var.")

    # Upload audio to GCS
    gcs_uri = _upload_to_gcs(audio_path, bucket_name)

    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=min_speakers,
        max_speaker_count=max_speakers,
    )

    config = speech.RecognitionConfig(
        language_code=language_code,
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hz,
        enable_automatic_punctuation=True,
        diarization_config=diarization_config,
        audio_channel_count=1,
    )
    audio = speech.RecognitionAudio(uri=gcs_uri)

    client = speech.SpeechClient()
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=3600)

    # Extract transcript and diarisation segments
    text_chunks: List[str] = []
    last_alt_words: List[speech.WordInfo] = []
    for result in response.results:
        alt = result.alternatives[0]
        text_chunks.append(alt.transcript)
        if alt.words:
            last_alt_words = alt.words

    segments = _group_words_into_segments(last_alt_words)
    full_text = " ".join(text_chunks).strip()

    if delete_after:
        try:
            storage.Client().bucket(bucket_name).blob(
                gcs_uri.split("/", 3)[-1]
            ).delete()
        except Exception:
            pass  # Non-fatal

    return {
        "text": full_text,
        "segments": segments,
        "full_response": response,
        "gcs_uri": gcs_uri,
    }
