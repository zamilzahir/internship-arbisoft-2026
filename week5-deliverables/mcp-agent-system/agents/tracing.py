"""
Tracing layer.

Every agent (supervisor, workers) and every MCP call (resource read / tool call)
writes a structured event through this Tracer. Events are:
  - printed live to stdout, and
  - appended as JSON lines to traces/trace.jsonl

so you get both a human-readable console trace and a machine-readable log you
can replay, diff, or feed into an evaluation harness later.
"""

import json
import os
import threading
import time
import uuid


class Tracer:
    def __init__(self, path: str = "traces/trace.jsonl"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._t0 = time.time()
        open(self.path, "w").close()  # fresh trace file per run

    def log(self, agent: str, event: str, name: str, payload: dict | None = None):
        entry = {
            "id": str(uuid.uuid4())[:8],
            "t": round(time.time() - self._t0, 4),
            "agent": agent,
            "event": event,
            "name": name,
            "payload": payload or {},
        }
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        preview = json.dumps(entry["payload"])[:80]
        print(f"  [{entry['t']:>7.3f}s] {agent:<14} {event:<20} {name:<14} {preview}")
        return entry

    def summary(self):
        with open(self.path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        by_agent = {}
        for e in lines:
            by_agent.setdefault(e["agent"], 0)
            by_agent[e["agent"]] += 1
        return {"total_events": len(lines), "events_by_agent": by_agent}
