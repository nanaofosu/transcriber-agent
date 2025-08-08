"""Configuration module for the transcription agent.

This module centralises the loading of environment variables used across the
project.  Secrets such as API keys are read from the process environment at
import time.  If an expected variable is not set, sensible defaults are
provided instead of raising an exception so that the application can still
initialise (for example, if summarisation is disabled).
"""

from __future__ import annotations

import os

#: OpenAI API key used for generating summaries.  If this value is empty
#: summaries will not be generated even when the `--summary` flag is used.
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

#: Path to the Google Cloud service account JSON key used by
#: :mod:`google_transcriber`.  The Google Speech client library uses
#: `GOOGLE_APPLICATION_CREDENTIALS` to locate credentials on disk.  If not
#: provided, the environment must be configured externally.
GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS", ""
)

#: Name of the Whisper model to load.  By default we use the largest model
#: available to maximise transcription quality.  This can be overridden to
#: reduce memory consumption or speed up inference.
WHISPER_MODEL_NAME: str = os.getenv("WHISPER_MODEL_NAME", "large")

__all__ = [
    "OPENAI_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "WHISPER_MODEL_NAME",
]
