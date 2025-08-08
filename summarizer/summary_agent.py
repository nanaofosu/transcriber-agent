from __future__ import annotations

import json
import regex as re
from typing import Dict, Any
from config import OPENAI_API_KEY  # type: ignore

openai = None  # lazy import


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_FIRST_OBJ = re.compile(r"(\{(?:[^{}]|(?R))*\})", re.S)  # recursive first JSON object


def _extract_json(text: str) -> Dict[str, Any] | None:
    """Return first JSON object found in text, handling ```json fences."""
    if not text:
        return None

    # 1) Prefer fenced ```json block
    m = _JSON_BLOCK.search(text)
    candidate = m.group(1) if m else None

    # 2) Otherwise first {...} blob
    if candidate is None:
        m2 = _FIRST_OBJ.search(text)
        candidate = m2.group(1) if m2 else None

    if not candidate:
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _bullet_fallback(raw: str) -> Dict[str, list[str]]:
    """Parse simple 'Key Takeaways:' / 'Action Items:' bullet lists if present."""
    key_takeaways: list[str] = []
    action_items: list[str] = []
    section = None
    for line in raw.splitlines():
        t = line.strip()
        if not t:
            continue
        if t.lower().startswith("key takeaways"):
            section = "kt"
            continue
        if t.lower().startswith("action items"):
            section = "ai"
            continue
        if t.startswith("- "):
            if section == "kt":
                key_takeaways.append(t[2:].strip())
            elif section == "ai":
                action_items.append(t[2:].strip())
    return {"key_takeaways": key_takeaways, "action_items": action_items}


def generate_summary(transcript: str) -> Dict[str, Any]:
    """Produce a structured summary with key_takeaways and action_items."""
    if not OPENAI_API_KEY or not transcript.strip():
        return {"key_takeaways": [], "action_items": []}

    global openai
    if openai is None:
        try:
            import openai as _openai  # type: ignore
            openai = _openai
        except ModuleNotFoundError:
            return {"key_takeaways": [], "action_items": []}

    openai.api_key = OPENAI_API_KEY

    system_prompt = (
        "You are a helpful assistant that summarizes meeting transcripts. "
        "Return ONLY valid JSON with exactly two keys: "
        "`key_takeaways` (list of strings) and `action_items` (list of strings)."
    )
    user_prompt = f"Summarize the following transcript:\n\n{transcript}"

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=800,
            # Hard request for JSON (prevents markdown fences in most cases)
            response_format={"type": "json_object"},  # <-- key line
        )

        raw_text = (response.choices[0].message.content or "").strip()

        # Try direct JSON first
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            # Strip ```json fences or pull first {...} if needed
            parsed = _extract_json(raw_text) or {}

        key_takeaways = parsed.get("key_takeaways", [])
        action_items = parsed.get("action_items", [])

        # Validate types; if invalid, try bullet fallback
        if not isinstance(key_takeaways, list) or not isinstance(action_items, list):
            fallback = _bullet_fallback(raw_text)
            key_takeaways = fallback["key_takeaways"]
            action_items = fallback["action_items"]

        # Final guardrails
        if not isinstance(key_takeaways, list):
            key_takeaways = []
        if not isinstance(action_items, list):
            action_items = []

        # Normalize to strings
        key_takeaways = [str(x).strip() for x in key_takeaways if str(x).strip()]
        action_items = [str(x).strip() for x in action_items if str(x).strip()]

        return {"key_takeaways": key_takeaways, "action_items": action_items}

    except Exception as e:
        print(f"⚠️ Summary generation failed: {e}")
        return {"key_takeaways": [], "action_items": []}
