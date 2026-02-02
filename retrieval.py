from pinecone_client import get_index, pc

index = get_index()


def hybrid_search(
    query: str,
    doc_ids: list[str] | None = None,
    dense_k: int = 50,
    rerank_k: int = 50,
    final_k: int = 5
):
    # ---- Embed query (dense + sparse) ----
    dense_q = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=query,
        parameters={"input_type": "query", "truncate": "END"}
    )[0]

    sparse_q = pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=query,
        parameters={"input_type": "query", "truncate": "END"}
    )[0]

    filter_clause = {"doc_id": {"$in": doc_ids}} if doc_ids else None

    # ---- Hybrid search ----
    results = index.query(
        namespace="ns1",  # no fixed namespace
        top_k=dense_k,
        vector=dense_q["values"],
        sparse_vector={
            "indices": sparse_q["sparse_indices"],
            "values": sparse_q["sparse_values"]
        },
        filter=filter_clause,
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": rerank_k,
            "rank_fields": ["chunk_text"]
        },
        include_metadata=True
    )

    matches = results["matches"]

    formatted = [
        (
            m["score"],
            m["metadata"]["chunk_text"],
            m["metadata"]
        )
        for m in matches[:final_k]
    ]

    return formatted


# =========================
# 🔴 ADDED: RETRIEVAL SIGNAL ANALYSIS
# =========================

def score_concentration(results, min_ratio=1.5):
    """
    Checks whether the top result dominates the rest.
    """
    if len(results) < 3:
        return False

    top_score = results[0][0]
    rest_scores = [s for s, _, _ in results[1:]]
    avg_rest = sum(rest_scores) / len(rest_scores)

    return top_score / (avg_rest + 1e-6) >= min_ratio


def chunk_agreement(results, min_overlap=0.1):
    """
    Measures agreement between top retrieved chunks.
    """
    texts = [text.lower() for _, text, _ in results[:5]]
    if len(texts) < 2:
        return False

    common_terms = set(texts[0].split())
    for t in texts[1:]:
        common_terms &= set(t.split())

    overlap_ratio = len(common_terms) / max(len(texts[0].split()), 1)
    return overlap_ratio >= min_overlap


def rerank_gap(results, min_gap=0.05):
    """
    Checks confidence gap between top reranked results.
    """
    if len(results) < 2:
        return False

    return (results[0][0] - results[1][0]) >= min_gap


def is_answerable(results):
    """
    Domain-agnostic, model-free answerability gate.
    """
    return (
        score_concentration(results)
        and chunk_agreement(results)
        and rerank_gap(results)
    )


# =========================
# EXISTING CODE (UNCHANGED)
# =========================

def assemble_chunks(
    results,
    allow_multi_section=False,
    max_chunks=4,
    min_score_ratio=0.6
):
    """
    results: list of (score, text, metadata), sorted desc
    """

    if not results:
        return "", []

    assembled_text = ""
    used_sources = []

    top_score = results[0][0]

    for score, text, meta in results:
        # Stop if chunk is much weaker than the top one
        if score < top_score * min_score_ratio:
            break

        assembled_text += text.strip() + "\n"
        used_sources.append(meta)

        if not allow_multi_section:
            # narrow query → only first strong chunk
            break

        if len(used_sources) >= max_chunks:
            break

    return assembled_text.strip(), used_sources


def needs_global_context(query: str) -> bool:
    q = query.lower()

    # Signals that the answer likely requires aggregation across sections/pages
    global_terms = [
        "limitations",
        "challenges",
        "stages",
        "roles",
        "implications",
        "advantages",
        "disadvantages",
        "why",
        "how does",
        "overall",
        "across",
        "compared to",
        "difference between",
    ]

    return any(term in q for term in global_terms)
