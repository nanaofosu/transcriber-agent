"""Output helpers for the transcription agent.

This package contains functions for formatting transcripts into various
file formats and for writing the resulting data to disk.  Consumers of
this package can import the desired functions directly or through the
package namespace.
"""

from .formatter import (
    format_plain_text,
    format_markdown,
    format_srt,
    format_docx,
)
from .save_output import save_output

__all__ = [
    "format_plain_text",
    "format_markdown",
    "format_srt",
    "format_docx",
    "save_output",
]
