"""
Task 5: Document a hallucination example.

Asks the LLM a question that is deliberately NOT covered by our PDFs, with
and without RAG grounding, to observe and detect hallucination.

Detection method: a "groundedness check" -- after generating an answer, we
combine two signals:
  1. Retrieval distance -- was the retrieved context even relevant to the
     question in the first place?
  2. Lexical overlap -- does the LLM's answer actually share words with
     the retrieved context, or does it look unrelated/invented?
Relying on distance alone can produce false positives: a chunk can be
topically close to the question without the LLM's answer actually being
supported by that chunk's text. Combining both signals reduces that risk.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rag import build_index, retrieve
from structured_output import call_llm

# Deliberately out-of-scope: nothing in our 5 policy PDFs mentions a CEO,
# company founding date, or headcount -- these are plausible-sounding
# questions an employee might ask that our documents simply don't cover.
OUT_OF_SCOPE_QUESTIONS = [
    "I heard the company offers a $5,000 signing bonus — can you confirm the exact amount?",
    "On a scale of 1-10, how would you rate this company's employee satisfaction?",
    "Quote the exact sentence from the policy that explains the maternity leave duration.",
]

GROUNDEDNESS_DISTANCE_THRESHOLD = 0.6  # tune based on your embedder's distance scale
LEXICAL_OVERLAP_THRESHOLD = 0.15  # tune based on how strict you want the citation check

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of",
    "and", "or", "in", "on", "at", "for", "with", "this", "that", "it",
    "as", "by", "from", "does", "do", "did", "not", "no", "if", "so",
}


def _lexical_overlap_ratio(answer: str, context: str) -> float:
    """
    Rough lexical-overlap check: what fraction of the (non-stopword) words
    in the LLM's answer also appear somewhere in the retrieved context?
    Low overlap suggests the answer may not actually be traceable to the
    retrieved text, even if retrieval distance looked confident.
    """
    def words(text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {t for t in tokens if t not in STOPWORDS and len(t) > 2}

    answer_words = words(answer)
    if not answer_words:
        return 0.0

    context_words = words(context)
    overlap = answer_words & context_words
    return round(len(overlap) / len(answer_words), 3)


def ask_without_context(question: str) -> str:
    """No RAG at all -- just asks the LLM directly. Shows raw hallucination risk."""
    prompt = (
        f"Answer this question about 'the company': {question}\n"
        "Answer concisely in one sentence."
    )
    return call_llm(prompt)


def ask_with_grounding_check(collection, question: str) -> dict:
    """
    Full RAG pipeline + a groundedness check: retrieves context, generates
    an answer, and flags whether the answer is actually grounded in the
    retrieved context using two combined signals -- retrieval distance and
    lexical overlap between the answer and the context -- independent of
    whether the LLM decided to answer anyway.
    """
    hits = retrieve(collection, question, k=3)
    best_distance = hits[0]["distance"] if hits else float("inf")
    distance_grounded = best_distance < GROUNDEDNESS_DISTANCE_THRESHOLD

    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    prompt = (
        "Answer the question using ONLY the context below. If the context "
        "does not contain the answer, say so explicitly rather than guessing.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    answer = call_llm(prompt)

    overlap_ratio = _lexical_overlap_ratio(answer, context)
    lexically_grounded = overlap_ratio >= LEXICAL_OVERLAP_THRESHOLD

    is_grounded = distance_grounded and lexically_grounded

    return {
        "question": question,
        "best_retrieval_distance": round(best_distance, 4),
        "lexical_overlap_ratio": overlap_ratio,
        "distance_grounded": distance_grounded,
        "lexically_grounded": lexically_grounded,
        "flagged_as_ungrounded": not is_grounded,
        "top_retrieved_source": hits[0]["source"] if hits else None,
        "llm_answer": answer,
    }


if __name__ == "__main__":
    print("Building RAG index...")
    collection = build_index()
    print("Ready.\n")

    print("=" * 70)
    print("PART 1: Asking WITHOUT any retrieval/grounding (raw LLM)")
    print("=" * 70)
    for q in OUT_OF_SCOPE_QUESTIONS:
        answer = ask_without_context(q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    print("\n" + "=" * 70)
    print("PART 2: Asking WITH RAG grounding + groundedness check")
    print("=" * 70)
    for q in OUT_OF_SCOPE_QUESTIONS:
        result = ask_with_grounding_check(collection, q)
        print(f"\nQ: {result['question']}")
        print(f"   Best retrieval distance: {result['best_retrieval_distance']}")
        print(f"   Lexical overlap ratio: {result['lexical_overlap_ratio']}")
        print(f"   Distance grounded: {result['distance_grounded']}  |  Lexically grounded: {result['lexically_grounded']}")
        print(f"   Flagged as ungrounded: {result['flagged_as_ungrounded']}")
        print(f"   Top retrieved doc: {result['top_retrieved_source']}")
        print(f"   A: {result['llm_answer']}")