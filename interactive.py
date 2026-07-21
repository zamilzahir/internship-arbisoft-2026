"""
interactive.py
---------------
Interactive CLI for the research agent - type your own questions instead
of running the fixed demo script.

Unlike demo.py (which feeds the agent four hardcoded questions to showcase
each Week 4 capability), this lets you drive the conversation yourself:
the agent keeps its session memory across turns, so you can ask a
follow-up question and it will recall earlier facts instead of
re-searching, exactly like it does in Step 3 of demo.py.

Run with:  python3 interactive.py
(requires GROQ_API_KEY and a search provider key in .env - see .env.example)

Commands while running:
  quit / exit / q   -> end the session
  log               -> print the tool-call log so far
"""

from agent import ResearchAgent


def main():
    agent = ResearchAgent()

    print("=" * 70)
    print("Research Agent - interactive mode")
    print("Type a question and press Enter. Type 'quit' to exit, "
          "'log' to see the tool-call log.")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        if user_input.lower() == "log":
            print("\n" + agent.tool_call_log())
            continue

        answer = agent.run(user_input)
        print("\nAgent:", answer)

    print(f"\nFull JSONL log at: logs/tool_calls.jsonl")


if __name__ == "__main__":
    main()
