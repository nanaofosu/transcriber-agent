"""Functions for writing formatted transcripts and summaries to disk.

These helpers encapsulate the logic for writing various output types to
files.  They also handle the optional saving of summary JSON objects
alongside the transcript.  All files are stored in a top‑level ``outputs``
directory which is created automatically if it does not exist.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

try:
    from docx.document import Document  # type: ignore
except ModuleNotFoundError:
    Document = None  # type: ignore


def save_output(
    content: Any,
    extension: str,
    original_audio_path: str,
    summary: Dict[str, Any] | None = None,
    output_dir: str = "outputs",
) -> None:
    """Write the transcript and optional summary to disk.

    Parameters
    ----------
    content:
        The formatted transcript to be written.  For ``docx`` files this
        should be a ``python-docx`` Document instance; otherwise a string.
    extension:
        File extension (without the dot) indicating the output format (e.g.
        ``"txt"``, ``"md"``, ``"srt"``, ``"docx"``).
    original_audio_path:
        Path to the input audio file.  The base name of this file is used
        when constructing the output filename.
    summary:
        An optional summary dictionary returned by
        :mod:`summarizer.summary_agent`.  When provided it will be written
        to a JSON file alongside the transcript.
    output_dir:
        Directory in which to create output files.  Defaults to ``outputs``.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(original_audio_path))[0]
    transcript_path = os.path.join(output_dir, f"{base_name}_transcript.{extension}")

    if extension == "docx":
        # Write out a Word document using the python-docx API.
        if Document is None:
            raise RuntimeError(
                "python-docx is not installed. Please install it to write DOCX files."
            )
        if not isinstance(content, Document):
            raise TypeError(
                "Content for DOCX output must be a python-docx Document instance"
            )
        content.save(transcript_path)
    else:
        # Assume the content is a text string.
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(str(content))

    # Write summary if provided.
    if summary:
        summary_path = os.path.join(output_dir, f"{base_name}_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
