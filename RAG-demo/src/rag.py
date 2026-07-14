"""
Task: Build a small RAG demo over a set of PDFs using ChromaDB.

Pipeline: query -> embed -> retrieve top-k chunks from ChromaDB -> build a
grounded prompt -> generate an answer.

Two interchangeable backends for the generation step:

  - "extractive": returns the most relevant retrieved sentence(s) verbatim
    as the answer, with the source cited. A legitimate, if unglamorous,
    RAG baseline -- kept as a zero-setup fallback.
  - "llm"        (default): sends the retrieved context to Groq via
    call_llm() (from structured_output.py) and returns a generated answer.
    A custom system instruction (RAG_SYSTEM_INSTRUCTION) is passed so the
    model responds in plain language instead of the JSON format used by
    structured_output.py's own pipeline. The prompt (build_prompt()) keeps
    the model grounded: it's instructed to answer ONLY from the retrieved
    context, and to say so explicitly if the context doesn't contain the
    answer, rather than guessing.
"""
from pathlib import Path
import chromadb

from ingest import load_and_chunk_all
from embeddings import LsaEmbedder  # LSA won the embedding comparison -- see outputs/observations.md
from structured_output import call_llm

DEFAULT_BACKEND = "llm"
TOP_K = 3

RAG_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that answers questions about company policy "
    "documents using only the provided context. Respond in plain, natural "
    "language sentences -- do not respond in JSON or any structured format."
)


def build_index():
    """
    Builds the ChromaDB collection from scratch on every call.

    Note: this deletes and recreates the "rag_demo" collection each time
    build_index() runs, rather than reusing an existing one. That's
    intentional for this demo -- it keeps the index guaranteed fresh and in
    sync with whatever's in data/pdfs/ -- but would be wasteful/expensive at
    scale with a large corpus. For a production setup, consider checking
    whether the collection already exists and is up to date before rebuilding.

    n_components=30: with too few latent dimensions, retrieval was
    collapsing unrelated topics together (e.g. password-related questions
    retrieving leave-policy or expense-policy content instead of IT
    security content). Raising this gives LSA more room to represent each
    topic distinctly. This value should stay in sync with the n_components
    used in embed_compare.py.
    """
    chunks = load_and_chunk_all()
    embedder = LsaEmbedder(n_components=30, max_features=2000)
    embedder.fit([c.text for c in chunks])

    client = chromadb.Client()
    try:
        client.delete_collection("rag_demo")
    except Exception:
        pass
    collection = client.create_collection(name="rag_demo", embedding_function=embedder)
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
    )
    return collection


def retrieve(collection, query: str, k: int = TOP_K):
    res = collection.query(query_texts=[query], n_results=k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "source": meta["source"], "distance": dist})
    return hits


def build_prompt(query: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    return (
        "Answer the question using ONLY the context below. If the context does not "
        "contain the answer, say so explicitly rather than guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def _generate_extractive(query: str, hits: list[dict]) -> str:
    """No-LLM fallback: surface the single most relevant chunk as the answer."""
    if not hits:
        return "No relevant context was retrieved for this question."
    best = hits[0]
    return f"(from {best['source']}): {best['text']}"


def _generate_with_llm(query: str, hits: list[dict]) -> str:
    """
    Sends the retrieved context to Groq via call_llm() and returns a
    generated answer in plain language (via RAG_SYSTEM_INSTRUCTION). The
    prompt (build_prompt) keeps the model grounded: it's instructed to
    answer only from the retrieved context, and to say so explicitly if the
    context doesn't contain the answer.
    """
    if not hits:
        return "No relevant context was retrieved for this question."
    prompt = build_prompt(query, hits)
    return call_llm(prompt, system_instruction=RAG_SYSTEM_INSTRUCTION)


def generate(query: str, hits: list[dict], backend: str = DEFAULT_BACKEND) -> str:
    if backend == "extractive":
        return _generate_extractive(query, hits)
    elif backend == "llm":
        return _generate_with_llm(query, hits)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def ask(collection, query: str, backend: str = DEFAULT_BACKEND, k: int = TOP_K):
    hits = retrieve(collection, query, k=k)
    answer = generate(query, hits, backend=backend)
    return {"query": query, "hits": hits, "answer": answer}


if __name__ == "__main__":
    print("Building index over PDFs in data/pdfs/ ...")
    collection = build_index()
    print("Index ready.\n")

    demo_questions = [
        "How many vacation days do new employees accrue?",
        "What is required before a remote employee can access the VPN?",
        "How far in advance must client entertainment expenses be approved?",
    ]

    print("Running demo questions:\n")
    for q in demo_questions:
        result = ask(collection, q)
        print(f"Q: {result['query']}")
        print(f"A: {result['answer']}")
        print(f"   (retrieved from: {[h['source'] for h in result['hits']]})\n")

    print("=" * 70)
    print("Now try your own question (type 'quit' to exit)")
    print("=" * 70)
    while True:
        user_question = input("\nYour question: ").strip()
        if user_question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not user_question:
            continue
        result = ask(collection, user_question)
        print(f"A: {result['answer']}")
        print(f"   (retrieved from: {[h['source'] for h in result['hits']]})")