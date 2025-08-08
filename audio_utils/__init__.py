"""Utility functions for processing and normalising audio.

This package contains helpers built on top of ffmpeg for converting
user‑supplied recordings into a format suitable for downstream transcription
models.  Functions here should not perform any transcription themselves but
may downsample, change channel count, strip metadata or perform other
preprocessing tasks.
"""

from .preprocess import preprocess_audio  # noqa: F401

__all__ = ["preprocess_audio"]
