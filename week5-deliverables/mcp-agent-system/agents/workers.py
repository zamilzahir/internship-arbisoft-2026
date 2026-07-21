"""
Worker sub-agents. The supervisor routes each incoming task to exactly one
of these. Every action a worker takes is written to the shared Tracer, so the
full call graph (supervisor -> worker -> tool) is reconstructable afterwards.
"""

import re

from .llm import HAVE_KEY, chat


class NotesWorker:
    """Handles note-taking tasks. Calls the MCP server's `add_note` tool."""

    name = "notes_worker"

    def __init__(self, tracer, mcp_client):
        self.tracer = tracer
        self.mcp_client = mcp_client

    async def handle(self, task: str) -> str:
        self.tracer.log(self.name, "worker_start", "handle", {"task": task})

        if HAVE_KEY:
            note_text = chat(
                system="Rewrite the user's request as a single concise note. "
                       "Reply with only the note text, no preamble.",
                user=task,
            )
        else:
            note_text = re.sub(
                r"^(add a note[: ]*|please jot down that[: ]*)", "", task, flags=re.I
            ).strip()

        result = await self.mcp_client.call_tool(
            "add_note", {"text": note_text}, caller=self.name
        )
        self.tracer.log(self.name, "worker_done", "handle", {"result": result})
        return result


class MathWorker:
    """Handles arithmetic tasks using a local `calculate` tool (not via MCP)."""

    name = "math_worker"

    def __init__(self, tracer):
        self.tracer = tracer

    async def handle(self, task: str) -> str:
        self.tracer.log(self.name, "worker_start", "handle", {"task": task})
        expr = self._extract_expression(task)

        self.tracer.log(self.name, "tool_call_start", "calculate", {"expr": expr})
        try:
            value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - sandboxed, digits/operators only
        except Exception as e:  # pragma: no cover
            value = f"error: {e}"
        self.tracer.log(self.name, "tool_call_end", "calculate", {"result": value})

        result = f"{expr} = {value}"
        self.tracer.log(self.name, "worker_done", "handle", {"result": result})
        return result

    @staticmethod
    def _extract_expression(task: str) -> str:
        # tip-style phrasing: "15% tip on $86.40"
        if "%" in task and re.search(r"\btip\b", task, re.I):
            pct = re.search(r"(\d+(\.\d+)?)\s*%", task)
            amounts = re.findall(r"\$?(\d+(\.\d+)?)", task)
            if pct and amounts:
                p = float(pct.group(1)) / 100
                a = float(amounts[-1][0])
                return f"{a} * {p}"
        # plain arithmetic expression, e.g. "42 * 17 + 8"
        tokens = re.findall(r"\d+\.\d+|\d+|[+\-*/()]", task)
        return "".join(tokens) if tokens else "0"
