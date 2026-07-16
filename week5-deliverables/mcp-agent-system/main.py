"""
End-to-end demo.

Boots the MCP server as a subprocess (over stdio), connects a client to it,
then runs a supervisor agent that routes a handful of tasks across two
sub-agents (NotesWorker, MathWorker) — one of which drives the MCP tool.
Every step is written to traces/trace.jsonl and printed live.

Usage:
    python3 main.py                      # mock LLM mode (no API key needed)
    ANTHROPIC_API_KEY=sk-... python3 main.py   # real Claude routes + writes notes
"""

import asyncio
import os

from agents.supervisor import Supervisor
from agents.tracing import Tracer
from client.mcp_client import MCPServerClient

TASKS = [
    "Add a note: schedule the client demo for Thursday at 3pm.",
    "What is 42 * 17 + 8?",
    "Please jot down that invoice #4521 was paid in full.",
    "Calculate 15% tip on a $86.40 bill.",
]


async def main():
    server_script = os.path.join(os.path.dirname(__file__), "server", "mcp_server.py")

    tracer = Tracer(path=os.path.join(os.path.dirname(__file__), "traces", "trace.jsonl"))

    mcp_client = MCPServerClient(server_script, tracer=tracer, agent_name="mcp_client")
    await mcp_client.connect()

    tools, resources = await mcp_client.list_capabilities()
    print(f"Connected. Server exposes tools={tools} resources={resources}\n")

    supervisor = Supervisor(tracer=tracer, mcp_client=mcp_client)

    for task in TASKS:
        print(f"=== TASK: {task}")
        result = await supervisor.handle(task)
        print(f"--> {result}\n")

    print("=== Final knowledge base ===")
    print(await mcp_client.read_resource("kb://notes"))

    print("\n=== Trace summary ===")
    print(tracer.summary())

    await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())
