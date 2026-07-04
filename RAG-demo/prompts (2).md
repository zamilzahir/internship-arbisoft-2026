# Prompts Log — Week 3

This log documents the prompts used with Claude while completing the Week 3
deliverables, including the actual code produced at each step.

---

## Entry 1

**Prompt:** "Build a small RAG demo over a set of PDFs using ChromaDB or
pgvector. Compare two embedding models and document the retrieval quality
along with your observations."

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Kick off Task 1 and
Task 2.

**Outcome:** Claude proposed a project structure (`data/pdfs/`, `src/`,
`outputs/`) and asked one clarifying question about LLM/API access and PDF
source before building anything.

---

## Entry 2

**Prompt:** "How do I start off the assignment?"

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Get a concrete plan
instead of starting blind.

**Outcome:** A 10-step ordered plan: scaffold project → generate sample data
→ chunk PDFs → embed with two models → compare retrieval → build RAG demo →
polish. Each step was mapped to a specific file to be created.

---

## Entry 3

**Prompt:** "Can you give me any raw data files? Bring me skeleton code."

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Get real starter
data and code, not just a plan.

**Outcome:** Claude generated `generate_sample_pdfs.py`, which builds 5
synthetic company-policy PDFs (remote work, leave, IT security, expense,
onboarding) with deliberately specific, checkable facts:

```python
DOCS = {
    "remote_work_policy.pdf": (
        "Remote Work Policy",
        [
            "Employees may work remotely up to 3 days per week...",
            "All remote employees must be reachable during core hours...",
            "The company provides a one-time home office stipend of $500...",
            "Remote employees are required to use company-issued VPN...",
        ],
    ),
    # ...4 more documents
}
```

---

## Entry 4

**Prompt:** "Bring skeleton code for embeddings, overlapping [chunks]."

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Get the chunking
logic (Task 1's core ingestion step).

**Outcome:** Claude produced `ingest.py`'s `chunk_text()` function:

```python
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60

def chunk_text(text, source, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    cleaned = " ".join(text.split())
    chunks = []
    start = 0
    idx = 0
    while start < len(cleaned):
        end = start + chunk_size
        piece = cleaned[start:end]
        if piece.strip():
            chunks.append(Chunk(
                id=f"{source}::chunk{idx}",
                text=piece,
                source=source,
                chunk_index=idx,
            ))
            idx += 1
        start += chunk_size - overlap
    return chunks
```

---

## Entry 5

**Prompt:** "What embedding models did I choose?" / "Bring the embedding
code."

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Get the two
embedding model implementations for Task 2's comparison.

**Outcome:** Claude built `embeddings.py` with two classes:

```python
class TfidfEmbedder(EmbeddingFunction):
    def __init__(self, max_features=2000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def fit(self, corpus):
        self.vectorizer.fit(corpus)
        self._fitted = True

    def __call__(self, input):
        matrix = self.vectorizer.transform(input).toarray()
        return matrix.tolist()


class LsaEmbedder(EmbeddingFunction):
    def __init__(self, n_components=100, max_features=2000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, corpus):
        n_components = min(self.svd.n_components, max(2, len(corpus) - 1))
        if n_components != self.svd.n_components:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.svd.fit(tfidf_matrix)
        self._fitted = True

    def __call__(self, input):
        tfidf_matrix = self.vectorizer.transform(input)
        reduced = self.svd.transform(tfidf_matrix)
        return reduced.tolist()
```

---

## Entry 6

**Prompt:** "Explain the code" (referring to `embed_compare.py`).

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Understand the
retrieval-scoring logic before running it.

**Outcome:** Claude walked through `evaluate()`'s core loop, which is what
actually computes precision@1/precision@3/MRR:

```python
def evaluate(collection, queries, k=3):
    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []
    for query, expected_source in queries:
        res = collection.query(query_texts=[query], n_results=k)
        retrieved_sources = [m["source"] for m in res["metadatas"][0]]
        rank = None
        for i, src in enumerate(retrieved_sources):
            if src == expected_source:
                rank = i + 1
                break
        if rank == 1:
            hits_at_1 += 1
        if rank is not None and rank <= 3:
            hits_at_3 += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    n = len(queries)
    return {
        "precision_at_1": round(hits_at_1 / n, 3),
        "precision_at_3": round(hits_at_3 / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
    }
```

---

## Entry 7

**Prompt:** "Bring me the RAG demo code."

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Get Task 1's actual
demo — retrieval plus an answer step.

**Outcome:** Claude built `rag.py`, including `build_index()` and
`retrieve()`:

```python
def build_index():
    chunks = load_and_chunk_all()
    embedder = LsaEmbedder(n_components=8, max_features=2000)
    embedder.fit([c.text for c in chunks])
    client = chromadb.Client()
    collection = client.create_collection(name="rag_demo", embedding_function=embedder)
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
    )
    return collection

def retrieve(collection, query, k=TOP_K):
    res = collection.query(query_texts=[query], n_results=k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "source": meta["source"], "distance": dist})
    return hits
```

---

## Entry 8

**Prompt:** "What does this mean?" (pointed at the chunk overlap math).

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Understand why
`start += chunk_size - overlap` produces overlapping windows.

**Outcome:** Claude explained the sliding-window mechanic with concrete
character positions: chunk 1 = chars 0–400, chunk 2 = chars 340–740 (since
400 − 60 = 340), so the last 60 characters of one chunk repeat as the first
60 of the next — preventing a fact from being cut exactly in half.

---

## Entry 9

**Prompt:** "Explain TF-IDF and LSA, very easy, with example."

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Understand the
conceptual difference between the two embedding models before comparing
them.

**Outcome:** Claude used a "librarian" analogy: TF-IDF only recognizes exact
keyword matches; LSA notices which words tend to appear together across
documents (e.g. "login" and "password"), so it can match a paraphrased
question to the right document even without shared vocabulary.

---

## Entry 10

**Prompt:** "How is precision determined? Explain the code."

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Verify
understanding of the scoring math, not just accept the numbers.

**Outcome:** Claude broke down precision@1 as a simple ratio: count how many
questions had the correct answer in exactly rank 1, divide by total
questions asked — walked through with a worked 5-question example (2/5 =
0.4).

---

## Entry 11

**Prompt:** "Why not precision@2?"

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Understand why
precision@1 and precision@3 specifically were chosen as metrics.

**Outcome:** Claude explained the metric choice should match how many
results the system actually uses downstream — since `rag.py` retrieves
`k=3` chunks, precision@3 measures what the system actually sees; there was
no `k=2` step in the pipeline, so no precision@2 was needed.

---

## Entry 12

**Prompt:** "So chunking should always be done into relevant topics?"

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Push back on an
overgeneralization before it became a factual claim in the write-up.

**Outcome:** Claude clarified that fixed-size chunking (used here) and
topic/section-aware chunking are a real tradeoff — topic-aware chunking
needs reliably detectable document structure, which real-world PDFs often
lack; fixed-size chunking is more robust but blind to natural boundaries.

---

## Entry 13

**Prompt:** "But we could have 5 PDFs content in 1 and make 5 chunks?"

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Explore an
alternative chunking granularity (1 chunk per document).

**Outcome:** Claude explained the tradeoff: 1 chunk per document is simpler
but makes retrieval trivially coarse (every question about a topic returns
the whole document, not the specific passage) — smaller chunks trade
simplicity for precision.

---

## Entry 14

**Prompt:** "Do you think it's relevant to RAG demo?" / "If it uses the PDFs
from the same package then let it be there."

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Decide project
folder structure for Tasks 3–5.

**Outcome:** Claude laid out two options (separate folder vs. same folder)
and, given the shared PDFs/venv, recommended keeping everything inside one
project folder rather than splitting across folders with import-path
workarounds.

---

## Entry 15

**Prompt:** "Let's focus on this task" / "How do I check?"

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Get concrete
terminal commands to actually run and verify the embedding comparison
locally.

**Outcome:** Claude provided the exact `cd` / `source venv/bin/activate` /
`python3 src/embed_compare.py` sequence and explained what output to expect
(both models tying on easy questions, LSA winning on hard/paraphrased ones).

---

## Entry 16

**Prompt:** "Make the observations.md easy and intuitive to understand at my
level."

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Rewrite the Task 2
write-up in first person, reflecting what was actually run and understood,
rather than Claude's own voice from the initial build.

**Outcome:** Claude rewrote `observations.md` removing first-person "I"
statements describing design decisions Claude made unilaterally, replacing
them with a plain-language version describing what was actually tested and
found.

---

## Entry 17

**Prompt:** "I want to change the folder name, not the repo name — which is
`week3-deliverables`."

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Rename the local
project folder to match the branch name convention.

**Outcome:** Claude gave the `git mv week3 RAG-demo` sequence (later
reverted back toward a broader name once Tasks 3–5 were added, since
"RAG-demo" undersold non-RAG work).

---

## Entry 18

**Prompt:** "How do I commit and push this to week3 deliverables?"

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Get Tasks 1–2 onto
GitHub properly, matching the existing branch pattern
(`feature/week2-deliverables`).

**Outcome:** Claude gave the full sequence: `git checkout main` → `git pull`
→ `git checkout -b feature/week3-deliverables` → `git add` → `git commit` →
`git push -u origin feature/week3-deliverables`.

---

## Entry 19

**Prompt:** "Make the whole thing" (Task 3: "Build a structured-output
pipeline: LLM → JSON → Pydantic validation → Save output").

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Build Task 3 end to
end.

**Outcome:** Claude created `schemas.py` with the `PolicyFact` Pydantic
model:

```python
class PolicyFact(BaseModel):
    policy_name: str = Field(...)
    category: str = Field(...)
    fact: str = Field(...)
    key_number: float | None = Field(None)
    source_grounded: bool = Field(...)

    @field_validator("category")
    @classmethod
    def category_must_be_short(cls, v):
        if len(v.split()) > 4:
            raise ValueError("category must be a short label (<=4 words)")
        return v

    @field_validator("fact")
    @classmethod
    def fact_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("fact cannot be empty")
        return v
```

---

## Entry 20

**Prompt:** (continuation of Entry 19) — full pipeline implementation.

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Wire the LLM call,
JSON parsing, validation, and retry logic together.

**Outcome:** Claude built `structured_output.py`'s `extract_fact()`, which
implements the full LLM → JSON → validate → retry chain:

```python
def extract_fact(passage, source_name, max_retries=MAX_RETRIES):
    prompt = f"Policy document: {source_name}\n\nPassage:\n{passage}\n\nExtract one fact as JSON."
    last_error = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            prompt = (
                f"Your previous response failed validation with this error:\n{last_error}\n\n"
                f"Please fix it and respond again with ONLY valid JSON..."
            )
        raw_response = call_llm(prompt)
        cleaned = _strip_markdown_fences(raw_response)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = f"Response was not valid JSON: {e}"
            continue
        try:
            return PolicyFact(**data)
        except ValidationError as e:
            last_error = str(e)
            continue
    raise ValidationError.from_exception_data(...)
```

---

## Entry 21

**Prompt:** (continuation of Entry 19) — validation tests for Task 4.

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Build tests that
intentionally fail on malformed schema output.

**Outcome:** Claude built `tests/test_schema_validation.py` with 14 tests,
including realistic malformed-LLM-output scenarios:

```python
def test_llm_returns_key_number_with_units_as_string(self):
    """LLM writes '15 days' instead of the plain number 15."""
    malformed = {
        "policy_name": "leave_policy.pdf",
        "category": "vacation",
        "fact": "Employees get 15 days of vacation.",
        "key_number": "15 days",
        "source_grounded": True,
    }
    with pytest.raises(ValidationError):
        PolicyFact(**malformed)
```

---

## Entry 22

**Prompt:** (continuation of Entry 19) — hallucination demo for Task 5.

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Build a script that
asks out-of-scope questions with and without grounding, plus a groundedness
check.

**Outcome:** Claude built `hallucination_demo.py`'s
`ask_with_grounding_check()`:

```python
def ask_with_grounding_check(collection, question):
    hits = retrieve(collection, question, k=3)
    best_distance = hits[0]["distance"] if hits else float("inf")
    is_grounded = best_distance < GROUNDEDNESS_DISTANCE_THRESHOLD
    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    prompt = (
        "Answer using ONLY the context below. If the context does not "
        "contain the answer, say so explicitly rather than guessing.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    answer = call_llm(prompt)
    return {
        "question": question,
        "best_retrieval_distance": round(best_distance, 4),
        "flagged_as_ungrounded": not is_grounded,
        "llm_answer": answer,
    }
```

---

## Entry 23

**Prompt:** "How do I get an API key? Can I use my Claude API?"

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Set up credentials
to run the live LLM calls in Tasks 3 and 5.

**Outcome:** Claude walked through creating an Anthropic API key, then
(given billing concerns) pivoted to a free-tier alternative — Google's
Gemini API via `console.anthropic.com` first, then `aistudio.google.com`
once cost became a concern.

---

## Entry 24

**Prompt:** [pasted a real, live API key directly into the chat, more than
once]

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** N/A (accidental
exposure, not an intentional request).

**Outcome:** Claude flagged the exposure each time, recommended revoking and
regenerating the key immediately, and explained the difference between an
API key's actual scope (limited to that one API) versus a general account
login, without repeating the exposed key back.

---

## Entry 25

**Prompt:** "Why is it not working / stuck?" (across several
`hallucination_demo.py` runs).

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Diagnose repeated
hangs and failures when calling the Gemini API.

**Outcome:** Claude isolated the problem with two timed tests — index
building alone (0.1s, fine) and a single API call alone (1.7s, fine) — then
traced the real failures to Gemini's free-tier daily quota (20
requests/day, `429 RESOURCE_EXHAUSTED`) and a transient `503 UNAVAILABLE`
server-overload error, both on Google's side.

---

## Entry 26

**Prompt:** "Do you suggest any other API instead of this?"

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Find a more
reliable provider after repeated Gemini quota/availability failures.

**Outcome:** Claude recommended Groq (no card required, fast inference,
generous free tier) and rewrote the LLM-calling code:

```python
def _get_client():
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found.")
    return Groq(api_key=api_key)

def call_llm(prompt):
    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
```

---

## Entry 27

**Prompt:** "Why does it keep bringing the older questions despite the fact
I updated it?"

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Resolve a mismatch
between the VS Code editor's contents and what the script actually ran.

**Outcome:** Claude diagnosed that edits weren't being saved to disk (VS
Code showed "2 unsaved" in the sidebar) and used a direct Python rewrite to
bypass the editor entirely:

```python
with open('hallucination_demo.py', 'r') as f:
    content = f.read()
old_block = '''OUT_OF_SCOPE_QUESTIONS = [...]'''
new_block = '''OUT_OF_SCOPE_QUESTIONS = [...]'''
if old_block in content:
    content = content.replace(old_block, new_block)
    with open('hallucination_demo.py', 'w') as f:
        f.write(content)
    print("SUCCESS: Questions updated.")
```

---

## Entry 28

**Prompt:** "I want a question like something it gives wrong result, what do
you suggest?"

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Find question
phrasings more likely to provoke a genuine hallucination, since direct
factual questions kept being correctly declined.

**Outcome:** Claude suggested leading/presupposing questions instead of open
ones — e.g. asserting a false stipend amount and asking the model to
"confirm" it, or asking for an "exact quote" of a policy sentence that
doesn't exist — since these are harder for a model to simply refuse than an
open factual question.

---

## Entry 29

**Prompt:** "Should I just answer that it didn't get hallucinated?"

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Decide how to
handle Task 5 when the pipeline kept correctly declining rather than
fabricating.

**Outcome:** Claude advised against submitting a null result as the final
answer, recommending either harder question phrasing or a supplementary
manual test in an external chat interface to actually observe and document
hallucination behavior, since the task requires a documented example, not
just a report that testing occurred.

---

## Entry 30

**Prompt:** "How do I push the updated files to GitHub?"

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Commit and push
Tasks 3–5 without leaking the `.env` file containing API keys.

**Outcome:** Claude found that `.gitignore` was not actually excluding
`.env` yet (confirmed via `git status | grep .env`), fixed it with
`echo ".env" >> .gitignore`, re-verified `.env` no longer appeared in `git
status`, then staged each intended file explicitly (never `git add .`) and
committed/pushed.

---

## Entry 31

**Prompt:** "Also, should we just delete the [API] key?"

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Consider whether
deleting an exposed key would resolve an unrelated code-not-saving issue.

**Outcome:** Claude clarified the two problems were unrelated — the file
not saving was a VS Code issue, not a credentials issue — and kept the key
rotation advice separate from the debugging steps already in progress.

---

## Entry 32

**Prompt:** "Can I give this to an external LLM as well?" (referring to
testing a hallucination-provoking question in ChatGPT/Claude/Gemini's chat
interface directly, outside the code).

**Tool:** Claude **Model:** Claude Sonnet 5 **Purpose:** Confirm whether
manually testing a prompt in a chat UI is a valid way to fulfill Task 5.

**Outcome:** Claude confirmed this was a legitimate approach for
*documenting* a hallucination example, while flagging that pasting internal
manager emails into external tools (as opposed to a generic test question)
was a separate consideration worth checking against company policy first.
