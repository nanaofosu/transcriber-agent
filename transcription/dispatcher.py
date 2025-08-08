"""Logic for selecting an appropriate transcription backend.

The :func:`transcribe` function defined here is the single entry point used
by the CLI and other modules to perform speech recognition.  It dispatches
requests to either the Whisper or the Google transcriber based on the
``use_diarization`` flag supplied by the caller.  Additional dispatch logic
could be added here to support other backends or heuristics in the future.
"""

from __future__ import annotations

from typing import Dict, Any

from .whisper_transcriber import transcribe_whisper
from .google_transcriber import transcribe_google


def transcribe(audio_path: str, use_diarization: bool = False) -> Dict[str, Any]:
    """Transcribe an audio recording.

    Parameters
    ----------
    audio_path:
        Path to the audio file to transcribe.
    use_diarization:
        When ``True`` the function will use the Google Speech‑to‑Text API with
        speaker diarisation enabled.  Otherwise it will use the Whisper model.

    Returns
    -------
    dict
        A dictionary containing at least the key ``text`` with the full
        transcript.  Whisper results additionally include ``segments`` and
        ``language``, whereas Google results include ``full_response``.
    """
    if use_diarization:
        return transcribe_google(audio_path)
    return transcribe_whisper(audio_path)
