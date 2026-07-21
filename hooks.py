"""
hooks.py
--------
A tool-call logging hook. Every tool invocation the agent makes (web
search, file read, etc.) is wrapped so we capture:
  - timestamp (ISO 8601, UTC)
  - tool name
  - input arguments
  - output (truncated) or error
  - duration in milliseconds

Logs go to logs/tool_calls.jsonl (one JSON object per line, easy to grep
or load with pandas) and are also mirrored to stdout so they're visible
live during a demo run.
"""

from __future__ import annotations
import json
import time
import functools
import datetime
from pathlib import Path
from typing import Callable

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "tool_calls.jsonl"


class ToolCallHook:
    """Central place tool calls are logged through, so demo.py / agent.py
    can also inspect history in-process (e.g. to print a summary table)."""

    def __init__(self, log_file: Path = LOG_FILE):
        self.log_file = log_file
        self.calls: list[dict] = []

    def record(self, name: str, args: dict, result: str | None,
               error: str | None, duration_ms: float):
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
                         .isoformat(timespec="milliseconds"),
            "tool": name,
            "args": args,
            "result_preview": (result[:300] + "...") if result and len(result) > 300 else result,
            "error": error,
            "duration_ms": round(duration_ms, 2),
        }
        self.calls.append(entry)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        status = "ERROR" if error else "OK"
        print(f"[hook] {entry['timestamp']} tool={name} status={status} "
              f"({entry['duration_ms']} ms)")
        return entry

    def summary(self) -> str:
        if not self.calls:
            return "No tool calls logged this session."
        lines = [f"{c['timestamp']}  {c['tool']:<15} "
                 f"{'ERROR' if c['error'] else 'OK':<5} {c['duration_ms']:>8.2f} ms"
                 for c in self.calls]
        return "\n".join(lines)


# Single shared hook instance used across the app.
hook = ToolCallHook()


def logged_tool(func: Callable) -> Callable:
    """Decorator that wraps a plugin function so every call is timed and
    logged through `hook`, regardless of how the tool is invoked."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        error = None
        result = None
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:  # noqa: BLE001 - we want to log then re-raise
            error = str(e)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            hook.record(func.__name__, kwargs or {"args": args}, result, error, duration_ms)

    return wrapper
