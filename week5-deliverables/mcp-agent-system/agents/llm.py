"""
Thin LLM wrapper shared by the supervisor and workers.

If ANTHROPIC_API_KEY is set in the environment, this calls the real Claude
API. Otherwise it falls back to a clearly-labeled deterministic mock so the
whole pipeline (routing -> workers -> MCP tool calls -> tracing) still runs
end-to-end without a key, e.g. for CI or a quick local demo.
"""

import os

HAVE_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")

_client = None
if HAVE_KEY:
    import anthropic

    _client = anthropic.Anthropic()


def chat(system: str, user: str, max_tokens: int = 200) -> str:
    """Single-turn text completion. Returns real Claude output, or a mock."""
    if not HAVE_KEY:
        return f"[mock-llm] {user}"

    resp = _client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text").strip()
