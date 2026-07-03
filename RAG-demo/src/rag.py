"""
Task: Build a small RAG demo over a set of PDFs using ChromaDB.

Pipeline: query -> embed -> retrieve top-k chunks from ChromaDB -> build a
grounded prompt -> generate an answer.

NOTE on the generation step: this sandbox has no route to any LLM API (no
key configured, and model-hosting domains are network-blocked -- see
embeddings.py for the same constraint on the embedding side). So `generate()`
below ships with two interchangeable backends:

  - "extractive"  (default, works with zero setup): returns the most
    relevant retrieved sentence(s) verbatim as the answer, with the source
    cited. This is a legitimate, if unglamorous, RAG baseline.
  - "llm"         (stub): drop your API call in `_generate_with_llm()` and
    switch DEFAULT_BACKEND to "llm". The retrieval half of the pipeline
    doesn't change at all.
"""
from pathlib import Path
import chromadb

from ingest import load_and_chunk_all
from embeddings import LsaEmbedder  # LSA won the embedding comparison -- see outputs/observations.md

DEFAULT_BACKEND = "extractive"
TOP_K = 3


def build_index():
    chunks = load_and_chunk_all()
    embedder = LsaEmbedder(n_components=8, max_features=2000)
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


def _generate_with_llm(prompt: str) -> str:
    """
    Plug a real LLM call in here, e.g.:

        import anthropic
        client = anthropic.Anthropic(api_key=...)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    Left unimplemented because this sandbox has no API key / no network route
    to an LLM provider.
    """
    raise NotImplementedError("Wire up an LLM client here to use backend='llm'.")


def generate(query: str, hits: list[dict], backend: str = DEFAULT_BACKEND) -> str:
    if backend == "extractive":
        return _generate_extractive(query, hits)
    elif backend == "llm":
        return _generate_with_llm(build_prompt(query, hits))
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

    for q in demo_questions:
        result = ask(collection, q)
        print(f"Q: {result['query']}")
        print(f"A: {result['answer']}")
        print(f"   (retrieved from: {[h['source'] for h in result['hits']]})\n")
