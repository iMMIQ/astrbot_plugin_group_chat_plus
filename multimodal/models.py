"""Domain contracts. Platform and AstrBot objects stay outside stored records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, NewType, TypedDict

from .routing import TurnRoute

RoomId = NewType("RoomId", str)
Message = dict[str, Any]


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image", "journal_image"]
    media_id: str


class MentionPart(TypedDict):
    type: Literal["mention"]
    target_id: str


class ReplyPart(TypedDict):
    type: Literal["reply"]
    event_id: str


class PokePart(TypedDict):
    type: Literal["poke"]
    actor_id: str
    target_id: str


class UnavailablePart(TypedDict):
    type: Literal["unavailable"]
    kind: str
    reason: str


class ProtocolPart(TypedDict):
    type: Literal["protocol"]
    messages: list[Message]


class ForwardNode(TypedDict):
    sender_id: str
    name: str
    parts: list[Part]


class ForwardPart(TypedDict):
    type: Literal["forward"]
    nodes: list[ForwardNode]
    available: bool


Part = (
    TextPart | ImagePart | MentionPart | ReplyPart | PokePart | UnavailablePart | ProtocolPart | ForwardPart
)


class Event(TypedDict):
    seq: int
    room: str
    event_id: str
    sender: str
    name: str
    kind: str
    received: float
    parts: list[Part]
    causal_anchor: int | None


class GenerationStatus(StrEnum):
    RUNNING = "running"
    GENERATED = "generated"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class DeliveryStatus(StrEnum):
    PENDING = "pending"
    SENT = "sent"
    PARTIAL = "partial"
    NONE = "none"
    UNCERTAIN = "uncertain"


@dataclass
class Selection:
    anchor: Event
    events: list[Event]
    protected: set[int]
    primary_media: set[str]
    reasons: dict[int, str]
    view_seq: int = 0
    chosen_media: set[str] = field(default_factory=set)
    chosen_events: list[Event] = field(default_factory=list)
    tokens: int = 0
    rebases: int = 0
    seed_seqs: set[int] = field(default_factory=set)
    frames: dict = field(default_factory=dict)
    canonical_current: list = field(default_factory=list)
    segment_id: str = ""
    rollover: str = ""
    scope: str = ""


@dataclass
class TurnState:
    selection: Selection
    gid: str
    provider: Any
    auto: bool = False
    route: TurnRoute | None = None
    snapshot: dict | None = None
    request: Any = None
    current: list[Part] = field(default_factory=list)
    baseline_protocol: list[Message] | None = None
    after_poke_attempted: bool = False
    receipts: list[Any] = field(default_factory=list)
    started: float = 0.0
    transport: Any = None
    run_context: Any = None
    public_messages: list[Message] = field(default_factory=list)
    media_leases: list[str] = field(default_factory=list)


def event_order(event: Event) -> tuple[int, int, int]:
    """A completed generation follows its trigger, even when inserted later."""
    parent = event.get("causal_anchor")
    return (parent if parent is not None else event["seq"], 1 if parent is not None else 0, event["seq"])
