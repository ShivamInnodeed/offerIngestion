"""Robust JSON extraction from LLM chat completion message content."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_START_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```\s*$", re.IGNORECASE)
_OPEN_THINK = "<" + "think>"
_CLOSE_THINK = "</" + "think>"
_OPEN_REDACTED = "<" + "redacted_reasoning>"
_CLOSE_REDACTED = "</" + "redacted_reasoning>"
_THINKING_BLOCK_RES = (
    re.compile(re.escape(_OPEN_THINK) + r".*?" + re.escape(_CLOSE_THINK), re.DOTALL | re.IGNORECASE),
    re.compile(
        re.escape(_OPEN_REDACTED) + r".*?" + re.escape(_CLOSE_REDACTED),
        re.DOTALL | re.IGNORECASE,
    ),
)


def extract_message_content(message: dict[str, Any] | None) -> str:
    if not message:
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("text") is not None:
                parts.append(str(part["text"]))
        return "\n".join(parts)
    for key in ("reasoning_content", "text", "output"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if content is not None:
        return str(content)
    return ""


def _strip_wrappers(text: str) -> str:
    raw = (text or "").strip().lstrip("\ufeff")
    if not raw:
        return ""
    for pattern in _THINKING_BLOCK_RES:
        raw = pattern.sub("", raw).strip()
    raw = _FENCE_START_RE.sub("", raw)
    raw = _FENCE_END_RE.sub("", raw)
    return raw.strip()


def _extract_balanced(text: str, *, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _as_dict(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                return item
    return None


def parse_llm_json(text: str | None) -> dict[str, Any]:
    cleaned = _strip_wrappers(text or "")
    if not cleaned:
        raise ValueError("LLM returned empty content")

    attempts: list[str] = [cleaned]
    for blob in (
        _extract_balanced(cleaned, open_ch="{", close_ch="}"),
        _extract_balanced(cleaned, open_ch="[", close_ch="]"),
    ):
        if blob and blob not in attempts:
            attempts.append(blob)

    last_error: Exception | None = None
    for candidate in attempts:
        try:
            parsed, _end = json.JSONDecoder().raw_decode(candidate)
            out = _as_dict(parsed)
            if out is not None:
                return out
        except json.JSONDecodeError as exc:
            last_error = exc
        try:
            out = _as_dict(json.loads(candidate))
            if out is not None:
                return out
        except json.JSONDecodeError as exc:
            last_error = exc

    preview = cleaned[:240].replace("\n", " ")
    if last_error is not None:
        raise ValueError(f"Could not parse JSON object from LLM content: {last_error}") from last_error
    raise ValueError(f"Could not parse JSON object from LLM content (preview={preview!r})")


def parse_llm_json_from_response(response_json: dict[str, Any]) -> dict[str, Any]:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = extract_message_content(message if isinstance(message, dict) else None)
    return parse_llm_json(content)
