"""
agent.py
--------
The research agent itself.

Pattern used: ReAct (Reason + Act), implemented via Claude's native tool
use. Each turn:
  1. Thought  -> the model reasons about what it needs (in its own text/
                 thinking before/around a tool call)
  2. Action   -> the model emits a tool_use block (web_search / read_file)
  3. Observation -> we execute the tool (through the logging hook),
                 and feed the result back as a tool_result block
  4. Repeat until the model responds with plain text (no more tool_use
     blocks), which is the final answer.

This loop is exactly what "multi-step reasoning" / ReAct means in
practice when you're using an LLM API with native tool-calling: the
API's tool_calls/tool-result protocol *is* the Thought/Action/Observation
cycle, so we don't need to hand-roll a "Thought:"/"Action:" text parser.

This version uses Groq's free API (OpenAI-compatible tool calling) instead
of a paid provider - get a free key at https://console.groq.com/keys.

Memory:
  - ConversationMemory keeps the raw transcript so the model has
    continuity turn to turn.
  - FactMemory stores atomic facts (extracted after each tool call and
    after each user message) so the agent can explicitly recall things
    from earlier in the session, independent of the model re-reading the
    whole transcript.
"""

from __future__ import annotations
import os
import json
import re
import time
from groq import Groq, BadRequestError, RateLimitError
from dotenv import load_dotenv

from memory import ConversationMemory, FactMemory
from hooks import hook
from plugins import web_search, read_file

load_dotenv()

# Free, tool-calling-capable model on Groq. gpt-oss-120b is Groq's own
# reference model for reliable tool calling (llama-3.3-70b-versatile was
# tried first but intermittently emits malformed tool calls). See other
# options at https://console.groq.com/docs/models
MODEL = "openai/gpt-oss-120b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use this for anything "
                "that requires up-to-date facts, news, or information you are "
                "not certain about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "num_results": {
                        "type": "integer",
                        "description": "How many results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a local .txt or .pdf file given its path. "
                "Use this when the user references a document or you need to "
                "look something up in a provided file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the .txt or .pdf file."},
                },
                "required": ["path"],
            },
        },
    },
]

TOOL_IMPLS = {
    "web_search": lambda **kw: web_search(**kw),
    "read_file": lambda **kw: read_file(**kw),
}

SYSTEM_PROMPT_TEMPLATE = """You are a careful research agent.

You have two tools:
- web_search: look up current information on the web
- read_file: read a local .txt or .pdf file

Follow a Reason -> Act -> Observe loop: think about what you need, call a
tool if you need external information, read the result, and either call
another tool or give a final answer. For multi-hop questions, break the
question into sub-steps and use tools for each step before answering -
do not guess at facts a tool could confirm.

Once you have gathered enough information to answer (usually 1-2 tool
calls per sub-question), STOP calling tools and give your final answer.
Do not repeat a search with only minor wording changes if an earlier
search already returned a usable result - use what you have.

Known facts from earlier in this session (use these instead of re-searching
if they already answer part of the question):
{facts_block}

Answer concisely and cite where a fact came from (web search vs. a file)
when it's relevant.
"""


class ResearchAgent:
    def __init__(self, api_key: str | None = None, model: str = MODEL):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model = model
        self.conversation = ConversationMemory()
        self.facts = FactMemory()

    # ---- memory helpers -------------------------------------------------

    def _remember_from_tool(self, tool_name: str, tool_input: dict, result: str):
        """Very lightweight fact extraction: store a compact summary of
        every tool result, tagged by source and turn, so it's recallable
        later without re-running the tool."""
        if tool_name == "web_search":
            key = f"search:{tool_input.get('query')}"
        else:
            key = f"file:{tool_input.get('path')}"
        summary = result if len(result) <= 500 else result[:500] + "..."
        self.facts.remember(key=key, value=summary, source=f"tool:{tool_name}",
                             turn=self.conversation.turn)

    def recall(self, query: str):
        """Public method so demo.py can show explicit memory recall."""
        return self.facts.recall(query)

    # ---- core loop --------------------------------------------------------

    def run(self, user_message: str, max_steps: int = 10, max_tool_calls: int = 5) -> str:
        """
        max_tool_calls hard-caps how many tool calls this run will make
        before forcing a text-only final answer. Some models (this one
        included) don't reliably follow a prompt instruction to "stop
        searching" - they'll keep calling tools until something forces
        them to stop, and even tool_choice="none" doesn't fully prevent
        it. Once the cap is hit, we drop the `tools` schema from the
        request entirely, so the model has nothing left to call and must
        answer in plain text - guaranteeing the loop terminates with an
        actual answer instead of hitting max_steps with nothing to show
        for it.
        """
        self.conversation.add_user(user_message)
        tool_call_count = 0

        for _ in range(max_steps):
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                facts_block=self.facts.as_context_block()
            )
            # Groq/OpenAI-style: system prompt is just the first message,
            # not a separate top-level argument.
            messages = [{"role": "system", "content": system_prompt}, *self.conversation.as_list()]

            force_final = tool_call_count >= max_tool_calls
            try:
                response = self._create_completion_with_retry(
                    messages, include_tools=not force_final
                )
            except BadRequestError as e:
                body = getattr(e, "body", None) or {}
                code = (body.get("error") or {}).get("code") if isinstance(body, dict) else None
                if code == "tool_use_failed":
                    # Even with no tools in the request, this model can
                    # still hallucinate tool-call-shaped output, which Groq
                    # rejects. Rather than loop forever, answer directly
                    # from whatever facts we've already gathered.
                    return self._fallback_answer_from_facts()
                raise

            choice = response.choices[0].message
            tool_calls = choice.tool_calls or []

            if not tool_calls:
                # Final answer: plain assistant text, no more tool use.
                self.conversation.add_assistant(choice.content)
                return choice.content or ""

            tool_call_count += len(tool_calls)

            # Record the assistant's tool-call request in the transcript
            # (required so the follow-up "tool" messages have something to
            # attach to), then execute every requested tool call.
            self.conversation.add_assistant(
                choice.content,
                tool_calls=[tc.model_dump() for tc in tool_calls],
            )

            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                try:
                    result = TOOL_IMPLS[tool_name](**tool_input)
                    self.conversation.add_tool_result(tc.id, result)
                    self._remember_from_tool(tool_name, tool_input, result)
                except Exception as e:  # noqa: BLE001
                    self.conversation.add_tool_result(tc.id, str(e), is_error=True)

        return "Reached max reasoning steps without a final answer."

    def _fallback_answer_from_facts(self) -> str:
        """Last-resort answer when the model can't produce a clean
        text-only response. Summarizes whatever facts were already
        gathered via tool calls this session, so the user still gets a
        useful result instead of a crash."""
        facts = self.facts.all_facts()
        if not facts:
            return ("I wasn't able to get a clean final answer from the model, "
                    "and no information was gathered yet either.")
        lines = ["I wasn't able to get a clean final answer from the model, "
                 "but here's what was found:"]
        for f in facts:
            lines.append(f"- {f}")
        return "\n".join(lines)

    def _create_completion_with_retry(self, messages, include_tools: bool = True, max_retries: int = 4):
        """
        Handles two known transient failure modes from Groq's API:

        1. groq.BadRequestError with code 'tool_use_failed' - the model
           occasionally emits a malformed tool call as raw text instead of
           proper structured JSON. Retrying almost always succeeds.
           Separately: even tool_choice="none" doesn't reliably stop this
           model from attempting a tool call, so when we want to force a
           final text answer, we omit the `tools` parameter from the
           request entirely - with no tool schema present at all, the
           model has nothing to call.
        2. groq.RateLimitError (429) - the free tier caps tokens-per-minute;
           a burst of requests (e.g. several tool calls in a row) can hit
           that cap. Groq's error message includes a suggested wait time
           (e.g. "Please try again in 9.24s"), which we parse and wait for.
        """
        kwargs = dict(
            model=self.model,
            max_tokens=1500,
            temperature=0.2,
            messages=messages,
        )
        if include_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"

        last_error = None
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**kwargs)
            except RateLimitError as e:
                if attempt >= max_retries - 1:
                    raise
                last_error = e
                wait_seconds = self._parse_retry_wait(e) or 10.0
                print(f"[agent] Rate limited, waiting {wait_seconds:.1f}s before retry "
                      f"({attempt + 1}/{max_retries})...")
                time.sleep(wait_seconds)
            except BadRequestError as e:
                body = getattr(e, "body", None) or {}
                code = (body.get("error") or {}).get("code") if isinstance(body, dict) else None
                if code == "tool_use_failed" and attempt < max_retries - 1:
                    last_error = e
                    time.sleep(0.5 * (attempt + 1))  # brief backoff before retry
                    continue
                raise
        raise last_error  # pragma: no cover - only reached if retries exhausted

    @staticmethod
    def _parse_retry_wait(error: RateLimitError) -> float | None:
        """Extract the suggested wait time from Groq's rate-limit message,
        e.g. '...Please try again in 9.24s...' -> 9.24. Adds a small buffer."""
        message = str(getattr(error, "message", None) or error)
        match = re.search(r"try again in ([\d.]+)s", message)
        if match:
            return float(match.group(1)) + 1.0  # small buffer past the suggested wait
        return None

    def tool_call_log(self) -> str:
        return hook.summary()