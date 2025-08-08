"""Command line interface for the transcription agent.

This module defines a helper function which sets up the argument parser
used by :mod:`main`.  Keeping argument parsing separate from application
logic makes it easier to test and potentially reuse in other contexts.
"""

from __future__ import annotations

import argparse


def parse_arguments() -> argparse.Namespace:
    """Return a populated :class:`argparse.Namespace` for the CLI.

    Defines supported flags and their defaults.  See the project README for
    usage examples.
    """
    parser = argparse.ArgumentParser(
        description="Local audio transcription agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the audio file to transcribe",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Enable speaker diarisation (use Google STT instead of Whisper)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate a summary using the OpenAI API",
    )
    parser.add_argument(
        "--output-format",
        default="txt",
        choices=["txt", "md", "srt", "docx"],
        help="Desired output file format for the transcript",
    )
    return parser.parse_args()
