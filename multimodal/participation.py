"""Participation policy, independent of framework hooks and persistence."""

from __future__ import annotations

import json
import random
import re
import time
from collections import deque

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _plain_text(parts) -> str:
    return "".join(p.get("text", "") for p in (parts or []) if p.get("type") == "text").strip()


class Participation:
    def __init__(self, config):
        self.config = config
        self.last_check = {}
        self.last_thread_check = {}
        self.last_nickname_check = {}
        self.last_bot_reply = {}
        self.last_react = {}
        self.last_plus_one = {}
        self.repeat_windows = {}

    def _cfg(self, cfg=None):
        return cfg if cfg is not None else self.config

    def candidate(self, room, cfg=None):
        c = self._cfg(cfg)
        return (
            c["auto_reply_enabled"]
            and self.ready(room, c)
            and random.random() < c["auto_candidate_probability"]
        )

    def ready(self, room, cfg=None):
        c = self._cfg(cfg)
        return time.monotonic() - self.last_check.get(room, -1e9) >= c["auto_reply_cooldown"]

    def claim(self, room, cfg=None):
        if not self.ready(room, cfg):
            return False
        self.last_check[room] = time.monotonic()
        return True

    def claim_signal(self, room, signal, cfg=None):
        c = self._cfg(cfg)
        now = time.monotonic()
        if signal == "thread_continue":
            min_gap = float(c.get("thread_min_interval_seconds", 4.0) or 0.0)
            if now - self.last_thread_check.get(room, -1e9) < min_gap:
                return False
            self.last_thread_check[room] = now
            return True
        if signal == "nickname_mention":
            cd = max(10.0, float(c.get("auto_reply_cooldown", 60)) / 3.0)
            if now - self.last_nickname_check.get(room, -1e9) < cd:
                return False
            self.last_nickname_check[room] = now
            return True
        if signal == "plus_one_break":
            self.last_check[room] = now
            return True
        return self.claim(room, c)

    def observe_member(self, room: str, sender: str, text: str, received: float | None = None):
        norm = " ".join((text or "").split())
        win = self.repeat_windows.setdefault(room, deque(maxlen=8))
        win.append((str(sender), norm, time.monotonic() if received is None else float(received)))

    def observe_bot_reply(
        self,
        room: str,
        partner_sender: str,
        anchor_seq: int,
        *,
        auto: bool = False,
        is_thread: bool = False,
    ):
        prev = self.last_bot_reply.get(room)
        if is_thread and prev and prev.get("partner") == str(partner_sender):
            streak = int(prev.get("streak", 0)) + 1
        elif auto:
            streak = 1
        else:
            streak = 0
        self.last_bot_reply[room] = {
            "ts": time.monotonic(),
            "partner": str(partner_sender),
            "seq": int(anchor_seq),
            "streak": streak,
        }
        if room in self.repeat_windows:
            self.repeat_windows[room].clear()

    def detect_plus_one(self, room: str, sender: str, text: str, cfg=None) -> str | None:
        c = self._cfg(cfg)
        if not c.get("plus_one_enabled", False):
            return None
        ignored = {str(x) for x in c.get("ignored_bot_ids", [])}
        if str(sender) in ignored:
            return None
        norm = " ".join((text or "").split())
        if (
            not norm
            or len(norm) > 30
            or norm.startswith(("/", "!", "！", "."))
            or "http://" in norm
            or "https://" in norm
        ):
            return None
        now = time.monotonic()
        cd = float(c.get("plus_one_cooldown", 180) or 0)
        last = self.last_plus_one.get(room)
        if last and now - last["ts"] < cd:
            return None
        if last and last.get("text") == norm and now - last["ts"] < max(cd, 600):
            return None
        min_users = max(2, int(c.get("plus_one_min_users", 3) or 3))
        win = self.repeat_windows.get(room)
        if not win or len(win) < min_users:
            return None
        distinct_senders = set()
        latest_ts = win[-1][2]
        for s, t, ts in reversed(win):
            if t != norm or abs(latest_ts - ts) > 90:
                break
            if s in ignored:
                continue
            distinct_senders.add(s)
        if len(distinct_senders) < min_users:
            return None
        p_repeat = max(0.0, min(1.0, float(c.get("plus_one_probability", 0.32))))
        p_break = max(0.0, min(1.0, float(c.get("plus_one_break_probability", 0.06))))
        roll = random.random()
        if roll < p_repeat:
            return "repeat"
        if roll < p_repeat + p_break:
            return "break"
        return None

    def mark_plus_one(self, room: str, text: str):
        self.last_plus_one[room] = {"ts": time.monotonic(), "text": " ".join((text or "").split())}
        if room in self.repeat_windows:
            self.repeat_windows[room].clear()

    def can_react(self, room: str, cfg=None) -> bool:
        c = self._cfg(cfg)
        if not c.get("emoji_react_enabled", False):
            return False
        cd = float(c.get("emoji_react_cooldown", 120) or 0)
        return time.monotonic() - self.last_react.get(room, -1e9) >= cd

    def mark_reacted(self, room: str):
        self.last_react[room] = time.monotonic()

    def classify_passive(self, room: str, anchor: dict, bot_id: str, cfg=None) -> str | None:
        c = self._cfg(cfg)
        if not c.get("auto_reply_enabled", False):
            return None
        sender = str(anchor.get("sender", ""))
        ignored = {str(x) for x in c.get("ignored_bot_ids", [])}
        if sender in ignored:
            return None
        parts = anchor.get("parts") or []
        if any(p.get("type") == "poke" for p in parts):
            return None
        bot_id = str(bot_id)
        mentions = [str(p.get("target_id", "")) for p in parts if p.get("type") == "mention"]
        has_reply = any(p.get("type") == "reply" for p in parts)
        txt = _plain_text(parts)
        txt_lower = txt.casefold()
        nicknames = [str(n).strip().casefold() for n in c.get("nicknames", []) if str(n).strip()]
        hit_nickname = bool(nicknames and any(n in txt_lower for n in nicknames))

        if mentions and bot_id not in mentions:
            return None
        if has_reply and not hit_nickname:
            return None

        now = time.monotonic()
        thread_win = float(c.get("thread_continuation_seconds", 0) or 0)
        max_turns = int(c.get("thread_max_turns", 3) or 0)
        min_gap = float(c.get("thread_min_interval_seconds", 4.0) or 0.0)
        last_bot = self.last_bot_reply.get(room)
        if (
            thread_win > 0
            and max_turns > 0
            and last_bot
            and last_bot.get("partner") == sender
            and now - last_bot["ts"] <= thread_win
            and int(last_bot.get("streak", 0)) < max_turns
            and now - self.last_thread_check.get(room, -1e9) >= min_gap
        ):
            return "thread_continue"

        if hit_nickname:
            nick_cd = max(10.0, float(c.get("auto_reply_cooldown", 60)) / 3.0)
            p_nick = max(0.0, min(1.0, float(c.get("nickname_candidate_probability", 0.7))))
            if now - self.last_nickname_check.get(room, -1e9) >= nick_cd and random.random() < p_nick:
                return "nickname_mention"

        if not self.ready(room, c):
            return None
        quiet_floor = min(20.0, float(c.get("auto_reply_cooldown", 60) or 0))
        if last_bot and quiet_floor > 0 and now - last_bot["ts"] < quiet_floor:
            return None
        base_prob = max(0.0, min(1.0, float(c.get("auto_candidate_probability", 0.02))))
        mult = 1.0
        has_image = any(p.get("type") == "image" for p in parts)
        if has_image or "?" in txt or "？" in txt:
            mult *= 1.5
        if not last_bot or (now - last_bot["ts"]) > 1800:
            mult *= 1.35
        eff_prob = min(1.0, max(base_prob, min(0.25, base_prob * mult)))
        if random.random() < eff_prob:
            return "ambient"
        return None

    @staticmethod
    def parse_gate(text) -> dict:
        if not isinstance(text, str):
            return {"action": "skip", "emoji": None}
        cleaned = _FENCE_RE.sub("", text.strip()).strip()
        try:
            value = json.loads(cleaned)
        except (ValueError, TypeError):
            return {"action": "skip", "emoji": None}
        if not isinstance(value, dict):
            return {"action": "skip", "emoji": None}
        action = value.get("action")
        if isinstance(action, str):
            act = action.strip().lower()
            if act == "reply":
                return {"action": "reply", "emoji": None}
            if act == "react":
                emoji = value.get("emoji")
                return {
                    "action": "react" if isinstance(emoji, str) and emoji.strip() else "skip",
                    "emoji": emoji.strip() if isinstance(emoji, str) else None,
                }
            return {"action": "skip", "emoji": None}
        if type(value.get("reply")) is bool:
            return {"action": "reply" if value["reply"] else "skip", "emoji": None}
        return {"action": "skip", "emoji": None}

    @staticmethod
    def accepts(text):
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            return False
        if not isinstance(value, dict):
            return False
        if type(value.get("reply")) is bool:
            return value["reply"]
        if isinstance(value.get("action"), str):
            return value["action"].strip().lower() == "reply"
        return False

    def cleanup(self):
        now = time.monotonic()
        cutoff = now - max(3600, self.config["auto_reply_cooldown"])
        self.last_check = {room: t for room, t in self.last_check.items() if t >= cutoff}
        self.last_thread_check = {room: t for room, t in self.last_thread_check.items() if t >= cutoff}
        self.last_nickname_check = {room: t for room, t in self.last_nickname_check.items() if t >= cutoff}
        self.last_react = {room: t for room, t in self.last_react.items() if t >= cutoff}
        self.last_bot_reply = {room: v for room, v in self.last_bot_reply.items() if v["ts"] >= cutoff}
        self.last_plus_one = {room: v for room, v in self.last_plus_one.items() if v["ts"] >= cutoff}
        self.repeat_windows = {
            room: w for room, w in self.repeat_windows.items() if w and now - w[-1][2] <= 600
        }
