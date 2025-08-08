"""Top‑level package exposing transcription functions.

This package defines a simple API for converting audio recordings into text.
Depending on whether speaker diarisation is requested, it will delegate to
either the Whisper or the Google transcriber.  Users of this package can
import :func:`transcription.transcribe` directly without worrying about which
backend is being used.
"""

from .dispatcher import transcribe  # noqa: F401

__all__ = ["transcribe"]
