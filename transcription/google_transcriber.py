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

from __future__ import annotations

import io
from typing import Dict, Any

try:
    from google.cloud import speech  # type: ignore
except ModuleNotFoundError:
    speech = None  # type: ignore

# `GOOGLE_APPLICATION_CREDENTIALS` is picked up automatically by the
# Google client library.  We import it here only for completeness.
from config import GOOGLE_APPLICATION_CREDENTIALS  # type: ignore  # noqa: F401


def transcribe_google(audio_path: str) -> Dict[str, Any]:
    """Transcribe an audio file using Google Speech‑to‑Text with diarisation.

    Parameters
    ----------
    audio_path:
        Path to the audio file.  WAV, FLAC or MP3 files are accepted.  For
        best results, audio should be mono and have a sampling rate of at
        least 8 kHz.

    Returns
    -------
    dict
        A dictionary containing the keys ``text`` and ``full_response``.  The
        ``text`` field contains the full transcript with diarisation tags
        concatenated.  The ``full_response`` field holds the raw API
        response returned by Google for further analysis or debugging.
    """
    if speech is None:
        raise RuntimeError(
            "google-cloud-speech is not installed. Please install the package to use Google transcription."
        )

    # Initialise the client.  The credentials are picked up from the
    # environment variable `GOOGLE_APPLICATION_CREDENTIALS` automatically.
    client = speech.SpeechClient()

    # Read the entire audio file into memory.  For long recordings or
    # streaming transcription you may wish to use a streaming API instead.
    with io.open(audio_path, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)

    # Enable diarisation.  If you know the number of speakers in advance,
    # ``diarization_speaker_count`` can be set.  Otherwise Google will
    # attempt to infer the number of speakers automatically.
    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=6,
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
        diarization_config=diarization_config,
    )

    # Invoke the API synchronously.  For long files consider using
    # ``long_running_recognize`` instead.
    response = client.recognize(config=config, audio=audio)

    # Assemble a simple concatenated transcript with speaker tags.  The
    # diarisation info is contained in the words of the last result.
    result_text_parts = []
    for result in response.results:
        alternative = result.alternatives[0]
        # Append the transcript directly if no speaker info is present.  For
        # final results with diarisation, the words field will be populated.
        if not alternative.words:
            result_text_parts.append(alternative.transcript)
        else:
            for word_info in alternative.words:
                speaker_tag = word_info.speaker_tag
                result_text_parts.append(f"[SPEAKER_{speaker_tag}] {word_info.word}")

    full_transcript = " ".join(result_text_parts)

    return {
        "text": full_transcript.strip(),
        "full_response": response,
    }
