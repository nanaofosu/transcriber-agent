"""Wrapper around the Whisper speech recognition model.

This module provides a convenience function for loading the OpenAI Whisper
model (large by default) and performing speech recognition on a given audio
file.  Results are returned as a dictionary compatible with the dispatcher
interface.
"""

from __future__ import annotations

from typing import Dict, Any

try:
    import whisper  # type: ignore
except ModuleNotFoundError:
    whisper = None  # type: ignore

# Import configuration from the project root.  When this module is loaded as
# part of the `transcription` package the root directory of the project is
# already on the Python path, so we can import `config` directly.
from config import WHISPER_MODEL_NAME  # type: ignore

# Module‑level cache so that the Whisper model is only loaded once.  Loading
# Whisper can be extremely time‑consuming and memory intensive, so we reuse
# the model across transcriptions rather than reloading on each call.
_MODEL = None  # type: ignore


def _load_model() -> whisper.Whisper:
    """Return a cached Whisper model, loading it if necessary.

    Raises a runtime error if the whisper library is not installed.
    """
    global _MODEL
    if whisper is None:
        raise RuntimeError(
            "The whisper library is not installed. Please install it to use Whisper transcriptions."
        )
    if _MODEL is None:
        _MODEL = whisper.load_model(WHISPER_MODEL_NAME)
    return _MODEL


def transcribe_whisper(audio_path: str) -> Dict[str, Any]:
    """Perform speech recognition on a file using Whisper.

    Parameters
    ----------
    audio_path:
        Path to the audio file to transcribe.  The file should already be in
        a format accepted by Whisper (usually WAV) and at an appropriate
        sample rate.  See :mod:`audio_utils.preprocess` for conversion
        helpers.

    Returns
    -------
    dict
        A dictionary containing the keys ``text``, ``segments`` and
        ``language``.  See the Whisper documentation for details of the
        returned fields.
    """
    model = _load_model()
    # We deliberately avoid passing any additional options here; users
    # requiring language hints or other parameters should extend this
    # function.  The ``transcribe`` method returns a dict with keys
    # including ``text``, ``segments`` and ``language``.
    result = model.transcribe(audio_path)
    return {
        "text": result.get("text", "").strip(),
        "segments": result.get("segments", []),
        "language": result.get("language"),
    }
