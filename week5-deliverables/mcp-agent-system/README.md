# MCP Supervisor / Worker Agent System

A minimal but complete example of:

1. **A custom MCP server** exposing one resource and one tool
2. **A custom MCP client** connecting to it
3. **A supervisor + worker agent graph** that routes tasks to ≥2 sub-agents
4. **A tracing layer** that logs every tool call across the whole graph

## Layout

```
mcp-agent-system/
├── server/
│   └── mcp_server.py     # MCP server: resource kb://notes, tool add_note
├── client/
│   └── mcp_client.py      # MCP client wrapper (stdio transport), traced
├── agents/
│   ├── tracing.py         # Tracer -> traces/trace.jsonl + live console log
│   ├── llm.py              # Claude API wrapper (real key, or mock fallback)
│   ├── supervisor.py       # Routes each task to notes_worker or math_worker
│   └── workers.py          # NotesWorker (uses MCP tool), MathWorker (local tool)
├── main.py                 # End-to-end demo, wires everything together
└── requirements.txt
```

## How the pieces fit together

```
 user task
    │
    ▼
 Supervisor.route()  ──► decides "notes" or "math"
    │
    ▼
 delegate to worker
    │
    ├── NotesWorker  ──► MCPServerClient.call_tool("add_note", …) ──► MCP server (subprocess, stdio)
    │
    └── MathWorker   ──► local calculate() tool
    │
    ▼
 every step above is also sent to Tracer.log(...)
    → printed live to stdout
    → appended to traces/trace.jsonl
```

The MCP server is launched as a subprocess and talked to over stdio (the
standard MCP transport) — this is the same mechanism Claude Code uses to
talk to MCP servers, so you can point Claude Code at `server/mcp_server.py`
directly instead of using the bundled client.

## Running it

```bash
pip install -r requirements.txt

# Mock mode — no API key needed. Routing is keyword-based, note text is
# extracted with regex, math is evaluated locally. Good for a quick sanity check.
python3 main.py

# Real mode — supervisor and NotesWorker actually call Claude.
export ANTHROPIC_API_KEY=sk-ant-...
python3 main.py
```

Every run prints a live trace like:

```
[  0.457s] supervisor     routing_decision     route          {"task": "...", "worker": "notes"}
[  0.457s] supervisor     delegate             notes          {"task": "..."}
[  0.458s] notes_worker   tool_call_start      add_note       {"text": "..."}
[  0.461s] notes_worker   tool_call_end        add_note       {"result": "Note added (#3): ..."}
```

and writes the full structured log to `traces/trace.jsonl` — one JSON object
per event (timestamp, agent, event type, name, payload), so you can replay or
analyze a run after the fact.

## Connecting Claude Code to the MCP server instead

Add this to your MCP config (e.g. `.mcp.json` / Claude Code settings):

```json
{
  "mcpServers": {
    "notes-kb": {
      "command": "python3",
      "args": ["/absolute/path/to/mcp-agent-system/server/mcp_server.py"]
    }
  }
}
```

Claude Code will then see the `add_note` tool and the `kb://notes` resource
directly.

## Extending it

- Add a 3rd worker (e.g. `ResearchWorker`) and update `Supervisor.route()`.
- Add more tools/resources to `mcp_server.py` — `@mcp.tool()` and
  `@mcp.resource()` are the only decorators you need.
- Swap the JSONL tracer for OpenTelemetry spans by changing `Tracer.log()` —
  the call sites elsewhere don't need to change.
