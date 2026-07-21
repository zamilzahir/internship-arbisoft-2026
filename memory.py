"""
memory.py
---------
Session memory for the research agent.

Two layers, on purpose:

1. ConversationMemory - the raw running transcript (what ReAct/tool-use
   needs turn to turn so the model has continuity).
2. FactMemory - a lightweight structured store of atomic facts the agent
   has learned (from tool results or from the user), each tagged with the
   turn it was learned on and a timestamp. This is what lets the agent
   "recall facts from earlier in the session" on demand, without having to
   re-scan the whole transcript, and lets us show a human-readable memory
   dump for debugging.

Both are in-memory only (cleared when the process exits) - that's enough
for "recall facts from earlier in the session," which is a same-session
requirement, not persistence across sessions.
"""

from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Fact:
    key: str
    value: str
    source: str          # e.g. "tool:web_search", "user", "tool:read_file"
    turn: int
    timestamp: float

    def __str__(self) -> str:
        return f"[{self.key}] {self.value} (from {self.source}, turn {self.turn})"


class ConversationMemory:
    """Holds the raw message list passed to the model each turn, in the
    OpenAI/Groq chat-completions message format (role: user/assistant/tool,
    with assistant tool_calls and matching role:"tool" results)."""

    def __init__(self):
        self.messages: list[dict[str, Any]] = []
        self.turn: int = 0

    def add_user(self, content):
        self.turn += 1
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str | None, tool_calls: list | None = None):
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str, is_error: bool = False):
        # Groq/OpenAI don't have a separate error flag on tool messages -
        # we just prefix the content so the model sees it plainly.
        text = f"Error: {content}" if is_error else content
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": text,
        })

    def as_list(self) -> list[dict[str, Any]]:
        return self.messages


class FactMemory:
    """
    Simple keyword-searchable fact store. Not a vector DB on purpose - for
    a session-scoped agent, exact/substring keyword match over a small
    number of facts is fast, transparent, and easy to demo/debug, which is
    the point of this assignment. Swap in embeddings later if the fact
    count grows large.
    """

    def __init__(self):
        self._facts: list[Fact] = []

    def remember(self, key: str, value: str, source: str, turn: int):
        self._facts.append(Fact(key=key, value=value, source=source,
                                 turn=turn, timestamp=time.time()))

    def recall(self, query: str, limit: int = 5) -> list[Fact]:
        """Return facts whose key or value contains any query token."""
        tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
        if not tokens:
            return []
        scored = []
        for f in self._facts:
            hay = f"{f.key} {f.value}".lower()
            score = sum(1 for t in tokens if t in hay)
            if score > 0:
                scored.append((score, f))
        scored.sort(key=lambda sf: sf[0], reverse=True)
        return [f for _, f in scored[:limit]]

    def all_facts(self) -> list[Fact]:
        return list(self._facts)

    def as_context_block(self) -> str:
        """Render all known facts as a block to inject into the system prompt."""
        if not self._facts:
            return "(no facts stored yet)"
        return "\n".join(f"- {f}" for f in self._facts)
