"""Remove generated transcript headers, preserving prose and code examples."""

from __future__ import annotations

import json

DECODER = json.JSONDecoder()


def _header(text, offset, primary=True):
    for marker, kind in (
        ("[message_metadata=", "metadata"),
        ("[native_bot_identity=", "identity"),
        ("[native_turn_control]", "control"),
        ("[戳一戳事件=", "poke"),
        ("[reply_to=", "reference"),
        ("[mention_id=", "reference"),
    ):
        if primary and kind == "reference":
            continue
        if not text.startswith(marker, offset):
            continue
        start = offset + len(marker)
        while start < len(text) and text[start].isspace():
            start += 1
        try:
            value, end = DECODER.raw_decode(text, start)
        except (ValueError, RecursionError):
            return None
        if kind == "metadata":
            fields = {"event_id", "sender_id", "name", "time", "source"}
            valid = (
                isinstance(value, dict)
                and fields <= value.keys()
                and all(isinstance(value[k], str) for k in fields)
                and value["source"] in {"self", "member", "quoted", "action"}
            )
        elif kind == "control":
            valid = (
                isinstance(value, dict)
                and {"bot_id", "anchor_event_id", "sender_id", "reason", "participation"} <= value.keys()
                and value["participation"] == "approved"
            )
        elif kind == "identity":
            valid = isinstance(value, dict) and isinstance(value.get("bot_id"), str)
        elif kind == "poke":
            valid = isinstance(value, dict) and all(
                isinstance(value.get(key), str) and bool(value[key]) for key in ("actor_id", "target_id")
            )
        else:
            valid = isinstance(value, str)
        if valid and kind == "control":
            return end
        while end < len(text) and text[end].isspace():
            end += 1
        if valid and text[end : end + 1] == "]":
            return end + 1
        return None
    return None


def strip_headers(text):
    """Strip complete internal header clusters only at unquoted line starts.

    Fenced/inline code, inline mentions of markers, malformed headers and
    ordinary JSON/logs are untouched. User input never goes through this filter.
    """
    output, fence, position = [], None, 0
    while position < len(text):
        newline = text.find("\n", position)
        line_end = newline + 1 if newline >= 0 else len(text)
        line = text[position:line_end]
        stripped = line.lstrip(" \t")
        if fence:
            output.append(line)
            if stripped.startswith(fence) and not stripped[len(fence) :].strip():
                fence = None
            position = line_end
            continue
        if stripped.startswith(("```", "~~~")):
            char = stripped[0]
            fence = char * (len(stripped) - len(stripped.lstrip(char)))
            output.append(line)
            position = line_end
            continue
        # CommonMark indented code examples are also literal data.
        if line.startswith(("    ", "\t")):
            output.append(line)
            position = line_end
            continue
        start = position + len(line) - len(stripped)
        end = _header(text, start)
        if end is None:
            output.append(line)
            position = line_end
            continue
        while True:
            offset = end
            while offset < len(text) and text[offset] in " \t":
                offset += 1
            following = _header(text, offset, primary=False)
            if following is None:
                # JSON may span several lines; consume the whole validated
                # header, then retain the ordinary remainder verbatim.
                newline = text.find("\n", offset)
                position = newline + 1 if newline >= 0 else len(text)
                remainder = text[offset:position]
                if remainder.strip():
                    output.append(remainder)
                break
            end = following
    return "".join(output)


def polish_reply_text(text: str, *, strip_trailing_period: bool = False) -> str:
    """Gently clean short casual chat lines without altering code or structured text."""
    if not text:
        return text
    if "```" in text or "~~~" in text:
        return text
    cleaned = text.strip()
    if (
        strip_trailing_period
        and cleaned
        and "\n" not in cleaned
        and len(cleaned) <= 25
        and "http://" not in cleaned
        and "https://" not in cleaned
        and cleaned.endswith("。")
        and not cleaned.endswith("。。")
        and cleaned.count("。") == 1
        and len(cleaned) > 1
    ):
        cleaned = cleaned[:-1]
    return cleaned


_STRUCTURED_PREFIXES = (
    "- ",
    "* ",
    "• ",
    "1. ",
    "1、",
    "# ",
    "## ",
    "### ",
    "📈",
    "📉",
    "💰",
    "🔍",
    "⚠️",
    "📊",
    "【",
)


def split_reply_bubbles(text: str, *, max_bubbles: int = 4, strip_trailing_period: bool = False) -> list[str]:
    """Split multi-line casual chat replies into separate short bubbles.

    Preserves fenced code blocks, structured lists/cards, and long prose as-is.
    """
    if not text or not text.strip():
        return [text] if text else []
    raw = text.strip()
    if "```" in raw or "~~~" in raw or len(raw) > 220:
        return [raw]
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) <= 1:
        return [polish_reply_text(raw, strip_trailing_period=strip_trailing_period)]
    if any(line.startswith(_STRUCTURED_PREFIXES) for line in lines):
        return [raw]
    cap = max(1, int(max_bubbles or 4))
    if len(lines) > cap:
        lines = lines[: cap - 1] + ["\n".join(lines[cap - 1 :])]
    return [
        polish_reply_text(seg, strip_trailing_period=strip_trailing_period) for seg in lines if seg.strip()
    ]
