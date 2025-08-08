"""Streamlit-based graphical interface for the transcription agent.

This application provides a simple web UI for uploading an audio or video file,
choosing transcription options and viewing or downloading the resulting
transcript.  It builds on top of the core modules in this project.

To run the application locally install the dependencies in
``requirements.txt`` and execute:

``
streamlit run ui/app.py
``
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Optional, Tuple

import streamlit as st

# Ensure that the project root is on sys.path so that imports work when
# launching this script via ``streamlit run ui/app.py``.  Without this,
# Python resolves imports relative to the ``ui`` directory, which would
# prevent modules like ``audio_utils`` from being found.
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio_utils.preprocess import preprocess_audio
from transcription import transcribe
from summarizer import generate_summary
from output.formatter import (
    format_plain_text,
    format_markdown,
    format_srt,
    format_docx,
)


def run_transcription(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    use_diarization: bool,
    generate_summ: bool,
    output_format: str,
) -> Tuple[str, Optional[bytes]]:
    """Handle the end-to-end transcription workflow for the uploaded file.

    Parameters
    ----------
    uploaded_file:
        The file object received from Streamlit's uploader.
    use_diarization:
        Whether to enable speaker diarisation (Google STT).
    generate_summ:
        Whether to generate a summary via OpenAI.
    output_format:
        The desired output format ('txt', 'md', 'srt', 'docx').

    Returns
    -------
    Tuple[str, Optional[bytes]]
        The formatted transcript and, for DOCX outputs, the binary data for
        downloading.  For other formats the second element will be ``None``.
    """
    # Persist uploaded file to a temporary location because the core
    # transcription modules expect a file path on disk.
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Preprocess audio to a normalised WAV file; this will also extract audio from videos.
    preprocessed_path = preprocess_audio(tmp_path)

    # Dispatch to the appropriate backend.
    result = transcribe(preprocessed_path, use_diarization=use_diarization)
    transcript = result.get("text", "")

    # Generate summary if requested.
    summary_dict = None
    if generate_summ:
        summary_dict = generate_summary(transcript)

    # Format transcript according to user selection.
    binary_data: Optional[bytes] = None
    if output_format == "txt":
        formatted = format_plain_text(transcript)
    elif output_format == "md":
        formatted = format_markdown(transcript)
    elif output_format == "srt":
        segments = result.get("segments", [])
        formatted = format_srt(segments) if segments else format_plain_text(transcript)
    elif output_format == "docx":
        doc = format_docx(transcript)
        buffer = io.BytesIO()
        doc.save(buffer)
        binary_data = buffer.getvalue()
        formatted = "(DOCX file generated)"
    else:
        formatted = transcript

    # Append summary to the displayed transcript for convenience.
    if summary_dict and (summary_dict.get("key_takeaways") or summary_dict.get("action_items")):
        formatted += "\n\n---\n**Summary**\n"
        if summary_dict.get("key_takeaways"):
            formatted += "\n**Key Takeaways:**\n"
            for item in summary_dict["key_takeaways"]:
                formatted += f"- {item}\n"
        if summary_dict.get("action_items"):
            formatted += "\n**Action Items:**\n"
            for item in summary_dict["action_items"]:
                formatted += f"- {item}\n"

    return formatted, binary_data


def main() -> None:
    st.set_page_config(page_title="Transcriber Agent", layout="wide")
    st.title("Transcriber Agent")
    st.write(
        "Upload an audio or video file and choose your transcription options. "
        "The transcript will be displayed below and can be downloaded."
    )

    uploaded_file = st.file_uploader(
        "Choose an audio or video file",
        type=["mp3", "wav", "flac", "m4a", "mp4", "mov", "mkv"],
        accept_multiple_files=False,
    )
    use_diarization = st.checkbox("Enable speaker diarisation (multi-speaker)")
    generate_summ = st.checkbox("Generate summary (requires OpenAI API key)")
    output_format = st.selectbox(
        "Output format",
        options=["txt", "md", "srt", "docx"],
        index=0,
    )

    if st.button("Transcribe"):
        if uploaded_file is None:
            st.error("Please upload a file first.")
        else:
            with st.spinner("Transcribing..."):
                formatted, binary_data = run_transcription(
                    uploaded_file,
                    use_diarization=use_diarization,
                    generate_summ=generate_summ,
                    output_format=output_format,
                )
            if output_format == "docx" and binary_data is not None:
                st.success("Transcription complete. Download your DOCX file below.")
                st.download_button(
                    label="Download DOCX",
                    data=binary_data,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            else:
                st.success("Transcription complete.")
                st.text_area(
                    "Transcript", formatted, height=400, help="The generated transcript."
                )
                st.download_button(
                    label="Download Transcript",
                    data=formatted,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.{output_format}",
                    mime="text/plain",
                )


if __name__ == "__main__":
    main()
