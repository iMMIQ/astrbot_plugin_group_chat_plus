"""Remove generated transcript headers, preserving prose and code examples."""

from __future__ import annotations

import json

DECODER = json.JSONDecoder()


def _header(text, offset, primary=True):
    for marker, kind in (
        ("[message_metadata=", "metadata"),
        ("[native_bot_identity=", "identity"),
        ("[native_turn_control]", "control"),
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
