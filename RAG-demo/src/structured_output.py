"""
Task 3: Structured-output pipeline.

Flow: LLM -> JSON -> Pydantic validation -> Save output.

Also implements a retry loop (Task-adjacent concept from the "remaining
Week 3 topics" list: output validation via schema validation + retry loops):
if the LLM's response fails Pydantic validation, we send it back with the
validation error and ask it to fix its own output, up to MAX_RETRIES times.

Requires GEMINI_API_KEY in a .env file (see README). Uses Google's Gemini
API since that's the key available for this project -- swap the
`call_llm()` internals for `anthropic` if using a Claude key instead; the
rest of the pipeline (parsing/validation/save/retry) doesn't change.
"""
import json
import os
import logging
from pathlib import Path
from pydantic import ValidationError
from dotenv import load_dotenv

from schemas import PolicyFact

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_RETRIES = 2

SYSTEM_INSTRUCTION = (
    "You extract a single fact from a company policy passage and respond with "
    "ONLY a JSON object -- no markdown fences, no commentary, no explanation. "
    "The JSON must have exactly these fields: "
    '"policy_name" (string), "category" (a short 1-4 word topic label), '
    '"fact" (one sentence stating the specific rule/number), '
    '"key_number" (the single most important number in the fact, as a plain '
    'number with no units -- or null if there is no single key number), '
    '"source_grounded" (true if the fact is directly supported by the '
    "passage given, false if you are inferring or guessing beyond it)."
)


def _get_client():
    """Lazily creates the Groq client so importing this module doesn't require a key."""
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


def call_llm(prompt: str) -> str:
    """Sends a prompt to Groq (Llama 3.3 70B), returns the raw text response."""
    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
    )
    if not response.choices:
        raise RuntimeError(
            "Groq API returned no choices — the response was empty or the request failed unexpectedly."
        )
    return response.choices[0].message.content


def _strip_markdown_fences(text: str) -> str:
    """LLMs often wrap JSON in ```json ... ``` even when told not to. Strip that if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return text.strip()


def extract_fact(passage: str, source_name: str, max_retries: int = MAX_RETRIES) -> PolicyFact:
    """
    The core pipeline: LLM -> JSON -> Pydantic validation -> (retry on failure) -> return.
    Raises ValidationError if still invalid after all retries.

    Logs every retry attempt (with the validation error that triggered it) so
    repeated LLM failures are visible for debugging, rather than being
    silently retried.
    """
    prompt = f"Policy document: {source_name}\n\nPassage:\n{passage}\n\nExtract one fact as JSON."
    last_error = None
    failed_attempts = []

    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.warning(
                f"[{source_name}] Retry attempt {attempt}/{max_retries} "
                f"after validation failure: {last_error}"
            )
            prompt = (
                f"Your previous response failed validation with this error:\n{last_error}\n\n"
                f"Please fix it and respond again with ONLY valid JSON for this passage:\n\n"
                f"Policy document: {source_name}\n\nPassage:\n{passage}"
            )

        raw_response = call_llm(prompt)
        cleaned = _strip_markdown_fences(raw_response)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = f"Response was not valid JSON: {e}. Raw response: {cleaned[:200]}"
            logger.warning(f"[{source_name}] Attempt {attempt}: JSON decode failed — {last_error}")
            failed_attempts.append({"attempt": attempt, "error": last_error})
            continue

        try:
            fact = PolicyFact(**data)
            if attempt > 0:
                logger.info(f"[{source_name}] Succeeded on retry attempt {attempt}")
            return fact  # success
        except ValidationError as e:
            last_error = str(e)
            logger.warning(f"[{source_name}] Attempt {attempt}: schema validation failed — {last_error}")
            failed_attempts.append({"attempt": attempt, "error": last_error})
            continue

    logger.error(
        f"[{source_name}] Failed after {max_retries + 1} attempts. "
        f"Full attempt history: {failed_attempts}"
    )
    raise ValidationError.from_exception_data(
        "PolicyFact", [{"type": "value_error", "loc": (), "msg": f"Failed after {max_retries + 1} attempts. Last error: {last_error}", "input": None}]
    )


def run_pipeline(passages: list[tuple[str, str]]) -> list[dict]:
    """Runs extract_fact over a list of (source_name, passage) pairs, saves results."""
    results = []
    for source_name, passage in passages:
        print(f"Extracting fact from {source_name}...")
        try:
            fact = extract_fact(passage, source_name)
            results.append({"status": "success", "source": source_name, "data": fact.model_dump()})
            print(f"  OK: {fact.fact}")
        except ValidationError as e:
            results.append({"status": "failed", "source": source_name, "error": str(e)})
            print(f"  FAILED validation after retries: {e}")

    out_path = OUTPUT_DIR / "structured_output_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    # Pull a few real passages from our PDFs to run the pipeline on.
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ingest import load_and_chunk_all

    chunks = load_and_chunk_all()
    # Use the first chunk from 3 different documents as a small demo batch.
    seen_sources = set()
    demo_passages = []
    for