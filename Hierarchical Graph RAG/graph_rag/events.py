from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GraphUpdateEvent:
    event_type: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)


GraphEventCallback = Callable[[GraphUpdateEvent], None]
