"""
demo.py
-------
End-to-end demonstration of every Week 4 capability in one run:

  1. Web search skill        -> agent searches the web for a current fact
  2. Session memory          -> agent recalls that fact later without re-searching
  3. Tool-call logging hook  -> every call above is timestamped in logs/tool_calls.jsonl
  4. File-read plugin        -> agent reads a local .txt/.pdf and uses its contents
  5. Multi-hop reasoning     -> a single question that requires chaining
                                 web_search -> read_file -> memory recall

Run with:  python demo.py
(requires ANTHROPIC_API_KEY and a search provider key in .env - see .env.example)
"""

from pathlib import Path
from agent import ResearchAgent

SAMPLE_FILE = Path(__file__).parent / "sample_notes.txt"


def ensure_sample_file():
    """Create a small sample .txt file so the demo works out of the box
    without requiring the user to supply their own document first."""
    if not SAMPLE_FILE.exists():
        SAMPLE_FILE.write_text(
            "Project Aurora - internal notes\n"
            "Budget: $120,000 for Q3\n"
            "Lead: Priya Nair\n"
            "Target launch: September 2026\n"
            "Primary risk: dependency on the new vendor API, still in beta.\n"
        )


def main():
    ensure_sample_file()
    agent = ResearchAgent()

    print("=" * 70)
    print("STEP 1: web search skill")
    print("=" * 70)
    answer1 = agent.run(
        "Search the web for the current version number of the Python "
        "programming language and tell me what it is."
    )
    print("\nAgent:", answer1)

    print("\n" + "=" * 70)
    print("STEP 2: file-read plugin")
    print("=" * 70)
    answer2 = agent.run(
        f"Read the file at {SAMPLE_FILE} and tell me who the project lead is "
        f"and what the Q3 budget is."
    )
    print("\nAgent:", answer2)

    print("\n" + "=" * 70)
    print("STEP 3: memory recall (no new tool call needed)")
    print("=" * 70)
    answer3 = agent.run(
        "Without searching again - what Python version did you find earlier, "
        "and what was Project Aurora's budget?"
    )
    print("\nAgent:", answer3)
    print("\n[explicit fact-memory lookup for 'budget']:")
    for fact in agent.recall("budget"):
        print(" -", fact)

    print("\n" + "=" * 70)
    print("STEP 4: multi-hop question chaining search + file + memory")
    print("=" * 70)
    answer4 = agent.run(
        f"Project Aurora's primary risk (see {SAMPLE_FILE}) is a dependency on "
        f"a vendor API that's still in beta, targeting a September 2026 launch. "
        f"Search the web for general best practices on managing vendor API "
        f"dependencies that are still in beta, then tell me: given the current "
        f"Python version you found earlier, the project's budget, and those "
        f"best practices, what are two concrete risk-mitigation steps you'd "
        f"recommend for Project Aurora?"
    )
    print("\nAgent:", answer4)

    print("\n" + "=" * 70)
    print("TOOL CALL LOG (from the hook)")
    print("=" * 70)
    print(agent.tool_call_log())
    print(f"\nFull JSONL log at: logs/tool_calls.jsonl")


if __name__ == "__main__":
    main()
