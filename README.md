# Research Agent — Week 4 Implementation

Implements all six Week 4 assignment items:

| # | Requirement | Where |
|---|---|---|
| 1 | Research agent with web-search skill (SerpAPI or Brave Search) | `plugins/web_search.py` |
| 2 | Memory to recall facts from earlier in the session | `memory.py` (`ConversationMemory` + `FactMemory`) |
| 3 | Hook that logs every tool call with timestamps | `hooks.py` (`@logged_tool`, writes `logs/tool_calls.jsonl`) |
| 4 | File-read plugin for `.txt` and `.pdf` | `plugins/file_read.py` |
| 5 | Multi-hop workflow demo | `demo.py` |
| 6 | Prompts log | `prompts.md` |

## How it fits together

- **`agent.py`** is the ReAct loop: it calls the Claude API with the
  `web_search` / `read_file` tool schemas, and on every `tool_use` block
  it executes the matching plugin, logs the call via the hook, feeds the
  `tool_result` back, and repeats until the model returns a plain-text
  final answer.
- **`memory.py`** keeps the raw conversation (for model continuity) and a
  separate `FactMemory` that stores a short summary of every tool result.
  Those facts are injected into the system prompt each turn, and can also
  be queried directly with `agent.recall("some keyword")`.
- **`hooks.py`** provides a `@logged_tool` decorator used by both plugins,
  so every call — success or failure — is timestamped and written to
  `logs/tool_calls.jsonl`, plus echoed to stdout during a run.
- **`plugins/`** contains the two skills: `web_search` (Brave Search or
  SerpAPI, switchable via `SEARCH_PROVIDER`) and `read_file` (`.txt` via
  plain read, `.pdf` via `pypdf`).

## Setup

```bash
cd research_agent
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env
# then edit .env and fill in:
#   GROQ_API_KEY   (free - get one at https://console.groq.com/keys)
#   SEARCH_PROVIDER=brave        (or serpapi)
#   BRAVE_API_KEY  (or SERPAPI_API_KEY)
```

This project uses **Groq's free API** (OpenAI-compatible tool calling) instead
of a paid LLM provider, so it costs nothing to run. Model used:
`llama-3.3-70b-versatile` — see other free options at
https://console.groq.com/docs/models.

## Run the demo

```bash
python demo.py
```

This will, in order:
1. Ask the agent a question requiring a web search.
2. Ask it to read a local sample file (`sample_notes.txt`, auto-created).
3. Ask it to recall both earlier facts from memory, without re-running
   any tools.
4. Ask a genuine multi-hop question that chains web search, the file's
   contents, and memory recall into one synthesized answer.
5. Print the full tool-call log produced by the hook.

Inspect `logs/tool_calls.jsonl` afterward to see every call with its
timestamp and duration.

## Use it interactively

```python
from agent import ResearchAgent

agent = ResearchAgent()
print(agent.run("What's the latest stable release of PostgreSQL?"))
print(agent.run("Read ./my_notes.pdf and summarize the action items."))
print(agent.run("Earlier you looked up a Postgres version - what was it?"))
```

## Notes / design decisions

- Tool-calling is done via Groq's **native (OpenAI-compatible) function
  calling** rather than a hand-rolled "Thought:/Action:" text parser — the
  `tool_calls` / `role: "tool"` message protocol already *is* the
  Reason→Act→Observe loop.
- Groq was chosen specifically because it's **free** and still supports
  proper tool calling (many free-tier LLM APIs don't).
- `FactMemory` uses simple keyword matching, not embeddings — for a
  session-scoped agent with a handful of facts, that's fast, transparent,
  and easy to debug/demo. Swap in a vector store if this needs to scale
  to long-running sessions with hundreds of facts.
- File reading is intentionally restricted to `.txt`/`.pdf` and is
  read-only, per the assignment scope.
