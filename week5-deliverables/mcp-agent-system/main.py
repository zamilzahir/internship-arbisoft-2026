"""
Interactive demo.

Boots the MCP server as a subprocess (over stdio), connects a client to it,
then lets you type tasks live, routed through a supervisor agent across two
sub-agents (NotesWorker, MathWorker) — one of which drives the MCP tool.
Every step is written to traces/trace.jsonl and printed live.

Usage:
    python3 main.py                      # mock LLM mode (no API key needed)
    GROQ_API_KEY=... python3 main.py     # real LLM routes + writes notes
"""

import asyncio
import os

from agents.supervisor import Supervisor
from agents.tracing import Tracer
from client.mcp_client import MCPServerClient


async def main():
    server_script = os.path.join(os.path.dirname(__file__), "server", "mcp_server.py")

    tracer = Tracer(path=os.path.join(os.path.dirname(__file__), "traces", "trace.jsonl"))

    mcp_client = MCPServerClient(server_script, tracer=tracer, agent_name="mcp_client")
    await mcp_client.connect()

    tools, resources = await mcp_client.list_capabilities()
    print(f"Connected. Server exposes tools={tools} resources={resources}\n")

    supervisor = Supervisor(tracer=tracer, mcp_client=mcp_client)

    print("Type a task (e.g. 'add a note: buy milk' or 'what is 12% of 340').")
    print("Type 'quit' or 'exit' to stop.\n")

    try:
        while True:
            task = input("> ").strip()
            if task.lower() in ("quit", "exit"):
                break
            if not task:
                continue

            result = await supervisor.handle(task)
            print(f"--> {result}\n")

    finally:
        print("=== Final knowledge base ===")
        print(await mcp_client.read_resource("kb://notes"))

        print("\n=== Trace summary ===")
        print(tracer.summary())

        await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())