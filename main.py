"""Entry point for the transcription agent.

This script wires together the CLI argument parsing, audio preprocessing,
transcription, summarisation, formatting and saving of output files.

To see available options run:

```
python main.py --help
```
"""

from __future__ import annotations

import os

from audio_utils import preprocess_audio
from transcription import transcribe
from summarizer import generate_summary
from output import (
    format_plain_text,
    format_markdown,
    format_srt,
    format_docx,
    save_output,
)
from ui import parse_arguments


def main() -> None:
    args = parse_arguments()

    audio_path = args.file
    # Preprocess the audio.  We convert to a WAV file with 16 kHz and mono
    # channels.  Whisper and Google can accept other formats but normalising
    # ensures consistent behaviour.
    preprocessed_path = preprocess_audio(audio_path)

    # Dispatch to the appropriate transcription backend based on the CLI flag.
    result = transcribe(preprocessed_path, use_diarization=args.multi)
    transcript = result.get("text", "")

    # Optionally generate a summary.  If summarisation is disabled or the
    # OpenAI API key is missing the summary will be empty.
    summary = None
    if args.summary:
        summary = generate_summary(transcript)

    # Choose the formatter based on the requested output format.
    if args.output_format == "txt":
        content = format_plain_text(transcript)
    elif args.output_format == "md":
        content = format_markdown(transcript)
    elif args.output_format == "srt":
        # SRT requires segment timing information.  Fallback gracefully if
        # segments are missing.
        segments = result.get("segments", [])
        content = format_srt(segments) if segments else format_plain_text(transcript)
    elif args.output_format == "docx":
        content = format_docx(transcript)
    else:
        raise ValueError(f"Unsupported output format: {args.output_format}")

    # Persist the transcript and (optionally) the summary to disk.
    save_output(
        content,
        args.output_format,
        original_audio_path=audio_path,
        summary=summary,
    )


if __name__ == "__main__":
    main()
