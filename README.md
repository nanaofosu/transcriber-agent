# Transcriber Agent

This repository contains a modular Python implementation of a local transcription
agent.  The system is designed to transcribe audio recordings, optionally
distinguish between multiple speakers, and generate concise summaries of the
content.  Outputs can be saved in a variety of formats, such as plain text,
markdown, subtitle (SRT) files and Word documents.

## Features

- **Whisper transcription** for single‑speaker audio using the large model.
- **Google Speech–to–Text** integration with speaker diarization for
  multi‑speaker recordings.
- Optional **summarisation** using the OpenAI Chat API to produce key
  takeaways and action items.
- Flexible **output formats** (TXT, MD, SRT, DOCX) with a unified API for
  formatting and saving files.
- Cleanly separated modules to allow easy extension (for example, a future
  graphical user interface).

## Structure

The repository is organised into a handful of top‑level packages:

```
transcriber-agent/
│
├── main.py                     # Command line entry point
├── config.py                   # API keys and model settings
├── requirements.txt            # External Python dependencies
│
├── audio_utils/
│   └── preprocess.py           # Audio conversion and cleanup
│
├── transcription/
│   ├── dispatcher.py          # Selects the appropriate transcriber
│   ├── whisper_transcriber.py  # Whisper wrapper
│   └── google_transcriber.py   # Google Cloud STT wrapper
│
├── summarizer/
│   └── summary_agent.py       # Generates key takeaways and actions
│
├── output/
│   ├── formatter.py           # Formatters for different file types
│   └── save_output.py          # Writes formatted data to disk
│
├── ui/
│   └── cli.py                 # CLI helper (optional)
│
└── samples/
    └── test_audio.mp3         # Placeholder audio for testing
```

## Usage

Run the entry script with at least the `--file` argument to specify the
audio file you wish to transcribe:

```sh
python main.py --file samples/test_audio.mp3
```

To transcribe a multi‑speaker recording and generate a summary saved as a
Word document:

```sh
python main.py --file interview.wav --multi --summary --output-format docx
```

Summaries are returned as structured JSON and written to disk alongside the
transcripts when requested.
