"""Google Cloud Speech–to–Text integration with speaker diarisation.

This module wraps the ``google.cloud.speech`` API to provide a simple
interface for transcribing audio and distinguishing between multiple
speakers.  Speaker diarisation assigns a unique speaker tag to each word
in the transcript, allowing conversations or interviews to be analysed.

Note that using this module requires a valid Google Cloud project and
service account key.  The path to your service account JSON file must be
set in the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable (or
configured externally) before calling :func:`transcribe_google`.
"""
"""Google Cloud Speech–to–Text integration with speaker diarisation (GCS + LRO)."""

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

# GOOGLE_APPLICATION_CREDENTIALS picked up automatically by the SDK.
# Imported only to make linting/tools aware that it's expected.
from config import GOOGLE_APPLICATION_CREDENTIALS  # type: ignore  # noqa: F401


def _upload_to_gcs(local_path: str, bucket_name: str, prefix: str = "uploads/") -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    # Give each upload a unique name to avoid collisions
    blob_name = f"{prefix}{uuid.uuid4().hex}_{os.path.basename(local_path)}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"


def _group_words_into_segments(words: List[speech.WordInfo]) -> List[Dict[str, Any]]:
    """
    Roll consecutive words with the same speaker_tag into segments.
    Each segment includes start/end (seconds as float), speaker, and text.
    """
    segments: List[Dict[str, Any]] = []
    if not words:
        return segments

    def ts_to_sec(ts) -> float:
        # ts is a google.protobuf.duration_pb2.Duration
        return ts.seconds + ts.nanos / 1e9 if ts is not None else 0.0

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
            # flush previous segment
            segments.append({
                "speaker": f"SPEAKER_{cur_speaker}",
                "start": seg_start or 0.0,
                "end": seg_end or seg_start or 0.0,
                "text": " ".join(cur_words).strip(),
            })
            # start new
            cur_speaker = spk
            cur_words = [w.word]
            seg_start = start
            seg_end = end

    # flush last
    if cur_words:
        segments.append({
            "speaker": f"SPEAKER_{cur_speaker}",
            "start": seg_start or 0.0,
            "end": seg_end or seg_start or 0.0,
            "text": " ".join(cur_words).strip(),
        })

    return segments


def transcribe_google(
    audio_path: str,
    *,
    gcs_bucket: Optional[str] = None,
    language_code: str = "en-US",
    sample_rate_hz: int = 16000,     # match your preprocess_audio output
    min_speakers: int = 2,
    max_speakers: int = 6,
    delete_after: bool = True,
) -> Dict[str, Any]:
    """Transcribe an audio file using Google STT with diarisation via LongRunningRecognize.

    Assumes ``audio_path`` is a mono LINEAR16 WAV at ``sample_rate_hz`` (e.g., from preprocess_audio).
    Uploads to GCS and passes a gs:// URI to the API to avoid 1-minute sync limits.

    Returns:
        {
          "text": str,
          "segments": [ {speaker, start, end, text}, ... ],
          "full_response": <raw google response>,
          "gcs_uri": "gs://bucket/object"
        }
    """
    if speech is None or storage is None:
        raise RuntimeError(
            "google-cloud-speech and google-cloud-storage are required. Install them to use Google transcription."
        )

    bucket_name = gcs_bucket or os.environ.get("GCS_BUCKET")
    if not bucket_name:
        raise RuntimeError("No GCS bucket specified. Set gcs_bucket param or GCS_BUCKET env var.")

    # 1) Upload to GCS
    gcs_uri = _upload_to_gcs(audio_path, bucket_name)

    # 2) Configure diarization
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
        # model="latest_long",  # optional; use your allowed model
        audio_channel_count=1,
    )

    audio = speech.RecognitionAudio(uri=gcs_uri)

    client = speech.SpeechClient()
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=3600)

    # 3) Build plain text and segments
    text_chunks: List[str] = []
    last_alt_words: List[speech.WordInfo] = []

    for result in response.results:
        alt = result.alternatives[0]
        text_chunks.append(alt.transcript)
        # Diarization words often only present in the final result,
        # but defensively collect from any populated alternative
        if alt.words:
            last_alt_words = alt.words

    segments = _group_words_into_segments(last_alt_words)
    full_text = " ".join(text_chunks).strip()

    # 4) (Optional) Delete uploaded blob
    if delete_after:
        try:
            storage.Client().bucket(bucket_name).blob(gcs_uri.split("/", 3)[-1]).delete()
        except Exception:
            # Non-fatal; leave the object if cleanup fails
            pass

    return {
        "text": full_text,
        "segments": segments,         # your UI can feed this to SRT or display
        "full_response": response,    # keep for debugging if needed
        "gcs_uri": gcs_uri,
    }
