"""
Supervisor agent: reads an incoming task, decides which worker should handle
it, delegates, and logs the routing decision + delegation to the tracer.
"""

import re

from .llm import HAVE_KEY, chat
from .workers import MathWorker, NotesWorker


class Supervisor:
    name = "supervisor"

    def __init__(self, tracer, mcp_client):
        self.tracer = tracer
        self.mcp_client = mcp_client

    async def route(self, task: str) -> str:
        self.tracer.log(self.name, "routing_start", "route", {"task": task})

        if HAVE_KEY:
            reply = chat(
                system=(
                    "You are a routing supervisor for a multi-agent system with two "
                    "workers: 'notes' (records/saves information) and 'math' "
                    "(performs a numeric calculation). Reply with exactly one word: "
                    "notes or math."
                ),
                user=task,
                max_tokens=5,
            )
            worker = "math" if "math" in reply.lower() else "notes"
        else:
            looks_mathy = bool(re.search(r"\d", task)) and bool(
                re.search(r"[+\-*/%]|\bcalculate\b|\btip\b", task, re.I)
            )
            worker = "math" if looks_mathy else "notes"

        self.tracer.log(self.name, "routing_decision", "route", {"task": task, "worker": worker})
        return worker

    async def handle(self, task: str) -> str:
        worker_name = await self.route(task)
        worker = (
            MathWorker(self.tracer)
            if worker_name == "math"
            else NotesWorker(self.tracer, self.mcp_client)
        )

        self.tracer.log(self.name, "delegate", worker_name, {"task": task})
        result = await worker.handle(task)
        self.tracer.log(self.name, "delegate_complete", worker_name, {"result": result})
        return result
