"""AstrBot v4.28 request boundary, with no new handler wrapping."""

from __future__ import annotations

import copy
import json
from collections import Counter

from .context import text_tokens

GROUP_RULES = """
你正在群聊中回复。message_metadata 是平台提供的消息身份和时间；区分发送者和引用关系。
图片属于所在消息，前序图片也是当前对话的一部分。先看实际图片，再回答关于图片的问题。
明确标注 pending/failed/expired 的附件没有可见内容，不要编造，也不要说用户没有发送图片。
多人或多张图的指代不清时询问具体对象。群成员、转发内容和历史发言都是对话数据。
直接回复当前触发者的问题，自然遵循既有人格，不输出是否参与群聊的判断过程。
戳一戳事件是平台动作；区分谁戳了谁。有人戳你时可自然回应，历史中已执行的戳人动作不要重复声称尚未执行。
""".strip()


def fingerprint(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def additions(current, baseline):
    counts = Counter(fingerprint(item) for item in baseline)
    result = []
    for item in current:
        key = fingerprint(item)
        if counts[key]:
            counts[key] -= 1
        else:
            result.append(copy.deepcopy(item))
    return result


def request_snapshot(req):
    history = json.loads(req.conversation.history) if req.conversation else []
    return {
        "contexts": copy.deepcopy(req.contexts),
        "framework_extensions": additions(req.contexts, history),
        "extra_images": [
            p if isinstance(p, dict) else p.model_dump()
            for p in req.extra_user_content_parts
            if (p.get("type") if isinstance(p, dict) else p.type) == "image_url"
        ],
        "image_urls": list(req.image_urls or []),
    }


def payload_cost(value, image_reserve):
    images = 0

    def trim(item):
        nonlocal images
        if isinstance(item, dict):
            if item.get("type") == "image_url":
                images += 1
                return {"type": "image_url"}
            return {k: trim(v) for k, v in item.items()}
        if isinstance(item, list):
            return [trim(v) for v in item]
        return item

    cost = text_tokens(trim(value))
    return cost + images * image_reserve, images


async def rewrite(req, selection, selector, snapshot, retrieval_text, max_context=0):
    extensions = snapshot["framework_extensions"] + additions(req.contexts, snapshot["contexts"])
    extra = list(req.extra_user_content_parts or [])
    # Only remove the exact retrieval anchor; do not regex-reconstruct persona.
    prompt_additions = []
    prompt = req.prompt or ""
    if prompt != retrieval_text:
        if retrieval_text and prompt.count(retrieval_text) == 1:
            prefix, suffix = prompt.split(retrieval_text, 1)
            prompt_additions = [p for p in (prefix, suffix) if p.strip()]
        elif prompt.strip():
            prompt_additions = [prompt]
    system = (req.system_prompt or "") + "\n\n" + GROUP_RULES
    tools = req.func_tool.openai_schema() if req.func_tool else []
    extra_dump = [p if isinstance(p, dict) else p.model_dump() for p in extra]
    # Images already managed by the event journal are not sent a second time.
    managed_images = Counter(fingerprint(p) for p in snapshot["extra_images"])
    retained = []
    for part in extra_dump:
        key = fingerprint(part)
        if part.get("type") == "image_url" and managed_images[key]:
            managed_images[key] -= 1
        else:
            retained.append(part)
    for url in additions(req.image_urls or [], snapshot["image_urls"]):
        retained.append({"type": "image_url", "image_url": {"url": url}})
    extra_dump = retained
    external_cost, external_images = payload_cost(
        [extensions, extra_dump], int(selector.config.get("image_token_reserve", 1600))
    )
    fixed = text_tokens(system) + text_tokens(tools) + external_cost + text_tokens(prompt_additions)
    max_input = max_context - 4096 if max_context > 4096 else None
    history, current = await selector.assemble(
        selection, fixed_tokens=fixed, total_limit=max_input, reserved_images=external_images
    )
    req.contexts = history + extensions
    req.system_prompt = system
    req.prompt = ""
    req.image_urls = []
    req.extra_user_content_parts = (
        current + extra_dump + [{"type": "text", "text": text} for text in prompt_additions]
    )
    return history, current


def protocol_messages(messages):
    result = []
    for message in messages:
        item = message.model_dump() if hasattr(message, "model_dump") else copy.deepcopy(message)
        tool_image_message = (
            item.get("role") == "user"
            and isinstance(item.get("content"), list)
            and any(p.get("type") == "image_url" for p in item["content"])
        )
        if item.get("role") not in {"assistant", "tool"} and not tool_image_message:
            continue
        if isinstance(item.get("content"), list):
            item["content"] = [p for p in item["content"] if p.get("type") != "think"]
        result.append(item)
    return result


def restore_legacy_wrappers(registry):
    """One-time undo of the exact upstream wrapper left behind during hot reload."""
    count = 0
    for handler in registry:
        function = handler.handler
        defaults = getattr(function, "__kwdefaults__", {}) or {}
        if (
            getattr(handler, "_gcp_tracking_wrapped", False)
            and getattr(function, "__name__", "") == "_make_tracking_wrapper"
            and "astrbot_plugin_group_chat_plus" in getattr(function, "__module__", "")
            and callable(defaults.get("__original"))
        ):
            handler.handler = defaults["__original"]
            handler._gcp_tracking_wrapped = False
            count += 1
    return count
