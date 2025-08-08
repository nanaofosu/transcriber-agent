"""Generate key takeaways and action items from a transcript.

This module uses the OpenAI Chat API (GPT‑4/GPT‑4o) to distil long
transcripts into a more digestible form.  It is designed to be optional and
will not function unless a valid API key is supplied in the environment.

The JSON structure returned contains two lists:

- ``key_takeaways``: high‑level points capturing the essence of the discussion.
- ``action_items``: specific follow‑up tasks or recommendations.
"""

from __future__ import annotations

import json
from typing import Dict, Any

# Import configuration from the project root.  When this module is run as part
# of the project the root directory will be on the Python path, so we can
# import ``config`` directly.
from config import OPENAI_API_KEY  # type: ignore

# Import openai lazily within ``generate_summary`` to avoid import errors
# during module import in environments where the package is not installed.
openai = None  # type: ignore


def generate_summary(transcript: str) -> Dict[str, Any]:
    """Produce a summary consisting of key takeaways and action items.

    Parameters
    ----------
    transcript:
        Full text of the audio transcription.

    Returns
    -------
    dict
        A structured summary with ``key_takeaways`` and ``action_items`` lists.

    Notes
    -----
    If the OpenAI API key is not set the function returns an empty summary.
    """
    # Return early if no API key is configured or the transcript is empty.
    if not OPENAI_API_KEY:
        return {"key_takeaways": [], "action_items": []}

    # Import openai lazily to avoid module import errors if the dependency is
    # missing.  If import fails, return an empty summary.
    global openai
    if openai is None:
        try:
            import openai as _openai  # type: ignore
            openai = _openai  # type: ignore
        except ModuleNotFoundError:
            return {"key_takeaways": [], "action_items": []}

    # Set the API key on the client.
    openai.api_key = OPENAI_API_KEY

    system_prompt = (
        "You are a helpful assistant that summarises meeting transcripts. "
        "Given the transcript, extract the key takeaways and action items. "
        "Respond in valid JSON with two arrays: 'key_takeaways' and 'action_items'."
    )
    user_prompt = (
        f"Transcript:\n{transcript}\n\n"
        "Please return a JSON object with the fields 'key_takeaways' and 'action_items' "
        "containing lists of bullet points summarising the discussion."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=512,
        )
        content = response["choices"][0]["message"]["content"]
    except Exception:
        # Gracefully handle any API errors by returning an empty summary.
        return {"key_takeaways": [], "action_items": []}

    # Attempt to parse the returned JSON.  If parsing fails, fall back to a
    # simple dictionary with empty lists.
    try:
        parsed = json.loads(content)
        key_takeaways = parsed.get("key_takeaways", [])
        action_items = parsed.get("action_items", [])
        if not isinstance(key_takeaways, list) or not isinstance(action_items, list):
            raise ValueError
        return {"key_takeaways": key_takeaways, "action_items": action_items}
    except Exception:
        return {"key_takeaways": [], "action_items": []}
