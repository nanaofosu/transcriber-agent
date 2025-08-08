"""Audio preprocessing helpers based on ffmpeg.

Before feeding audio into a speech recognition model, it is often necessary to
ensure that the recording has a consistent sampling rate, channel count and
encoding.  The functions in this module leverage the ffmpeg command‑line
utility (via the ``ffmpeg-python`` bindings) to perform these conversions.

The default behaviour is to convert any input audio file into a single
channel 16 kHz WAV file.  These settings follow common recommendations for
speech recognition engines such as Whisper and Google Speech‑to‑Text.

``ffmpeg`` must be installed on the host system for these functions to work.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    # Import ffmpeg lazily.  Some environments (such as test runners)
    # may not have ffmpeg installed; delaying the import prevents
    # import errors unless the preprocessing function is actually called.
    import ffmpeg  # type: ignore
except ModuleNotFoundError:
    ffmpeg = None  # type: ignore


def preprocess_audio(
    input_path: str,
    *,
    output_path: Optional[str] = None,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> str:
    """Convert an arbitrary audio file into a normalised WAV recording.

    Parameters
    ----------
    input_path:
        Path to the user‑supplied audio file.  Supported formats depend on
        ffmpeg but typically include MP3, WAV, FLAC, M4A and many others.
    output_path:
        Optional path where the converted file should be written.  If not
        supplied, the output will be created alongside the input file with
        a ``.wav`` extension appended.
    sample_rate:
        Sampling rate in Hertz.  Whisper expects 16 kHz by default.  Google
        supports a range of sample rates but 16 kHz is also common.
    channels:
        Number of audio channels.  Setting this to ``1`` produces mono audio
        which is required by certain recognition engines.

    Returns
    -------
    str
        The absolute path to the converted file.

    Notes
    -----
    This function will overwrite ``output_path`` if it already exists.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    # Derive an output filename if none was provided.
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}.wav"

    # Build and execute the ffmpeg command.  We explicitly map the audio
    # stream into WAV with the desired sample rate and channel count.  The
    # ``overwrite_output`` flag allows ffmpeg to replace existing files.
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg-python is not installed. Please install ffmpeg and the "
            "ffmpeg-python package or skip preprocessing."
        )
    (
   ffmpeg.input(input_path).audio
        .output(
            output_path,
            format="wav",
            ac=channels,
            ar=sample_rate,
        )
        .run(overwrite_output=True, quiet=True)
    )

    return os.path.abspath(output_path)
