# Prompts.md — How This Project Was Built

This is a reference guide: for each file in `mcp-agent-system/`, here's the
prompt you'd give an AI assistant (like Claude) to generate it, plus the
follow-up questions/answers that shaped the final version. Use this to
regenerate any single file from scratch, extend the project, or as a
template for building something similar.

---

## 1. `server/mcp_server.py`

**Prompt:**
> "Build a custom MCP server in Python exposing exactly one resource and one
> tool. Use FastMCP. The resource should return the current contents of a
> simple in-memory notes list. The tool should let a caller add a new note
> to that list and return a confirmation string."

**Follow-up Q&A:**
- *Q: What should the resource URI be?* → `kb://notes`
- *Q: Does the tool need input validation?* → Keep it simple — just accept
  a `text: str` argument.
- *Q: Should notes persist across restarts?* → No, in-memory is fine for
  this demo.

---

## 2. `client/mcp_client.py`

**Prompt:**
> "Write a Python MCP client wrapper class that connects to a local MCP
> server over stdio, and exposes `read_resource(uri)` and
> `call_tool(name, arguments)` methods. Every call should also be logged
> through a shared tracer object if one is provided."

**Follow-up Q&A:**
- *Q: What transport should it use?* → stdio (same transport Claude Code
  uses), via `StdioServerParameters` + `stdio_client`.
- *Q: Should the tracer be required?* → No — make it optional so the
  client can be used without tracing too.

---

## 3. `agents/tracing.py`

**Prompt:**
> "Add a tracing layer that logs every tool call and resource read across
> the whole agent graph — supervisor, workers, and the MCP client. Log to
> both the console (live, human-readable) and a JSONL file, so I can
> replay a run later."

**Follow-up Q&A:**
- *Q: What fields per event?* → timestamp, agent name, event type, target
  name, and a payload dict.
- *Q: One trace file per run, or append forever?* → Fresh file each run
  (`traces/trace.jsonl` gets truncated on `Tracer()` init).

---

## 4. `agents/llm.py`

**Prompt:**
> "Write a small LLM wrapper that calls the real Anthropic API if
> `ANTHROPIC_API_KEY` is set in the environment, and otherwise falls back
> to a clearly-labeled mock response — so the rest of the pipeline still
> runs end-to-end with zero cost and no API key."

**Follow-up Q&A:**
- *Q: Which model?* → `claude-sonnet-5` by default, overridable via
  `CLAUDE_MODEL` env var.
- *Q: What should the mock return?* → Just echo the input prefixed with
  `[mock-llm]` so it's obvious in logs when mock mode is active.

---

## 5. `agents/workers.py`

**Prompt:**
> "Create two worker agent classes: a `NotesWorker` that turns a task into
> a note and saves it via the MCP `add_note` tool, and a `MathWorker` that
> extracts an arithmetic expression from the task and evaluates it locally.
> Both should log every step to the tracer, and both should work with or
> without a real LLM available."

**Follow-up Q&A:**
- *Q: How does MathWorker parse expressions?* → Regex extraction of
  digits/operators, plus a special case for "X% tip on $Y" phrasing.
- *Q: Should eval() be sandboxed?* → Yes — call it with
  `{"__builtins__": {}}` so nothing but arithmetic can run.

---

## 6. `agents/supervisor.py`

**Prompt:**
> "Write a supervisor agent that looks at an incoming task and decides
> whether to route it to the notes worker or the math worker. Use the real
> LLM to decide if a key is available, otherwise use keyword/regex
> heuristics. Log the routing decision and delegation to the tracer."

**Follow-up Q&A:**
- *Q: What if the LLM reply is ambiguous?* → Default to `notes` unless the
  word "math" appears in the reply.
- *Q: Can it route to more than one worker per task?* → Not currently —
  one worker per task, kept simple on purpose. (Good extension point.)

---

## 7. `main.py`

**Prompt:**
> "Write an end-to-end demo script that boots the MCP server as a
> subprocess, connects the client, runs a supervisor over a handful of
> sample tasks (mix of note-taking and math), prints the final knowledge
> base contents, and prints a trace summary at the end."

**Follow-up Q&A:**
- *Q: How many sample tasks?* → 4 — two note tasks, two math tasks
  (including one that needs light parsing, like a tip calculation).
- *Q: Should it close the connection cleanly?* → Yes — use an
  `AsyncExitStack` and `await mcp_client.close()` at the end.

---

## 8. `requirements.txt`

**Prompt:**
> "List the exact pip dependencies needed for an MCP server + client and
> the Anthropic Python SDK."

**Answer:**
```
mcp>=1.28.0
anthropic>=0.117.0
```

---

## 9. `README.md`

**Prompt:**
> "Write a README for this project: explain the folder layout, how the
> pieces connect (a small ASCII diagram is nice), how to run it in both
> mock mode and real mode, and how to point Claude Code at the MCP server
> directly instead of using the bundled client."

**Follow-up Q&A:**
- *Q: Should it explain how to extend the project?* → Yes — a short
  "Extending it" section with 2–3 concrete next steps (add a 3rd worker,
  add more tools/resources, swap the tracer backend).

---

## How to use this file

- **Rebuilding a single file:** copy its prompt above into a new chat and
  paste in any neighboring files it imports from, for context.
- **Extending the project:** write a new section here first (prompt +
  Q&A) before generating the code — it keeps the "why" documented
  alongside the "what."
- **Teaching / demoing:** walk through this file top to bottom instead of
  the code — it explains the *decisions*, not just the implementation.
