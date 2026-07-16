"""
Custom MCP server.

Exposes:
  - ONE resource: kb://notes            -> current contents of an in-memory notes KB
  - ONE tool:     add_note(text: str)   -> appends a note and returns confirmation

Run standalone for debugging:
    python3 server/mcp_server.py
(Normally it's launched over stdio by an MCP client, not run directly by a human.)
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("notes-kb-server")

# --- in-memory "database" -------------------------------------------------
_notes: list[str] = [
    "Project kickoff meeting is scheduled for Monday.",
    "Remember to review the Q3 budget draft.",
]


# --- the one resource -------------------------------------------------------
@mcp.resource("kb://notes")
def get_notes() -> str:
    """Return every note currently stored in the knowledge base."""
    if not _notes:
        return "No notes yet."
    return "\n".join(f"{i + 1}. {n}" for i, n in enumerate(_notes))


# --- the one tool -------------------------------------------------------
@mcp.tool()
def add_note(text: str) -> str:
    """Add a new note to the knowledge base.

    Args:
        text: the note content to store.
    """
    _notes.append(text)
    return f"Note added (#{len(_notes)}): {text}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
