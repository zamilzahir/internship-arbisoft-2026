"""
Task: Compare two embedding models and document retrieval quality.

Indexes the same chunk set into two separate ChromaDB collections (one per
embedding model), runs an identical set of test queries with known
ground-truth source documents against both, and reports precision@1,
precision@3, and MRR for each model.
"""
import json
from pathlib import Path
import chromadb

from ingest import load_and_chunk_all
from embeddings import TfidfEmbedder, LsaEmbedder

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Ground-truth eval set: query -> the source PDF that should be retrieved.
TEST_QUERIES = [
    ("How many vacation days do employees get in their first two years?", "leave_policy.pdf"),
    ("What is the password rotation requirement for IT security?", "it_security_policy.pdf"),
    ("How much is the home office stipend for remote employees?", "remote_work_policy.pdf"),
    ("What is the daily meal reimbursement limit for international travel?", "expense_policy.pdf"),
    ("How long is the new hire onboarding program?", "onboarding_policy.pdf"),
    ("What happens if an employee fails two phishing simulations in a row?", "it_security_policy.pdf"),
    ("How many weeks of parental leave does the secondary caregiver get?", "leave_policy.pdf"),
    ("What is the cap on client entertainment expenses without pre-approval?", "expense_policy.pdf"),
]

# A harder set: same underlying facts, but paraphrased to minimize shared
# vocabulary with the source text. This is where a purely lexical model
# (TF-IDF) is expected to struggle relative to a model that captures latent
# semantic/co-occurrence structure (LSA).
HARD_PARAPHRASED_QUERIES = [
    ("Do new employees get help paying for internet at home?", "remote_work_policy.pdf"),
    ("How many paid days off can I take if a close relative dies?", "leave_policy.pdf"),
    ("What happens to my database access if I don't renew it in time?", "it_security_policy.pdf"),
    ("Am I allowed to fly business class on a short domestic trip?", "expense_policy.pdf"),
    ("Who do I get paired with when I first join the company?", "onboarding_policy.pdf"),
]


def build_collection(client, name: str, embedder, chunks):
    embedder.fit([c.text for c in chunks])
    # Fresh collection each run.
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name=name, embedding_function=embedder)
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
    )
    return collection


def evaluate(collection, queries, k=3):
    """Returns per-query results plus aggregate precision@1, precision@3, MRR."""
    results = []
    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []

    for query, expected_source in queries:
        res = collection.query(query_texts=[query], n_results=k)
        retrieved_sources = [m["source"] for m in res["metadatas"][0]]
        retrieved_distances = res["distances"][0]

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

        results.append({
            "query": query,
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "retrieved_distances": [round(d, 4) for d in retrieved_distances],
            "rank_of_correct": rank,
        })

    n = len(queries)
    metrics = {
        "precision_at_1": round(hits_at_1 / n, 3),
        "precision_at_3": round(hits_at_3 / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
    }
    return results, metrics


def main():
    print("Loading and chunking PDFs...")
    chunks = load_and_chunk_all()
    print(f"  {len(chunks)} chunks from {len(set(c.source for c in chunks))} documents\n")

    client = chromadb.Client()

    models = {
        "tfidf": TfidfEmbedder(max_features=2000),
        "lsa": LsaEmbedder(n_components=8, max_features=2000),
    }

    all_results = {}
    for model_name, embedder in models.items():
        print(f"Indexing with {model_name}...")
        collection = build_collection(client, f"chunks_{model_name}", embedder, chunks)

        easy_results, easy_metrics = evaluate(collection, TEST_QUERIES, k=3)
        print(f"  [easy/lexical queries]   precision@1={easy_metrics['precision_at_1']}  "
              f"precision@3={easy_metrics['precision_at_3']}  mrr={easy_metrics['mrr']}")

        hard_results, hard_metrics = evaluate(collection, HARD_PARAPHRASED_QUERIES, k=3)
        print(f"  [hard/paraphrased queries] precision@1={hard_metrics['precision_at_1']}  "
              f"precision@3={hard_metrics['precision_at_3']}  mrr={hard_metrics['mrr']}\n")

        all_results[model_name] = {
            "easy_queries": {"per_query": easy_results, "metrics": easy_metrics},
            "hard_paraphrased_queries": {"per_query": hard_results, "metrics": hard_metrics},
        }

    with open(OUTPUT_DIR / "embedding_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results written to {OUTPUT_DIR / 'embedding_comparison_results.json'}")

    return all_results


if __name__ == "__main__":
    main()
