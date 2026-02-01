
from answering import rewrite_with_llm, format_citations

from retrieval import hybrid_search,needs_global_context,assemble_chunks



gsq= [
    {
        "id": "Q1",
        "question":"What are the main components, objectives, and architectural models of a computer network, and how do peer-to-peer and client/server architectures differ in terms of advantages and limitations?",
        "expected_sections": ["Introduction"],
    }]

    # {
    #     "id": "Q2",
    #     "question": "At which stages of ADHD assessment is artificial intelligence applied?",
    #     "expected_sections": ["Introduction"],
    # }
def run_inspection(top_k=3):
    for q in gsq:
        print("\n" + "=" * 90)
        print(f"🧪 QUESTION {q['id']}")
        print("Q:", q["question"])

        retrieved = hybrid_search(
            query=q["question"],
            final_k=top_k
        )

        print("\n--- Retrieved Chunks (after hybrid + rerank) ---")
        for i, (score, doc, meta) in enumerate(retrieved[:top_k], 1):
            print(f"\n[{i}] Score: {score:.3f}")
            print(f"Section: {meta.get('section')} | Page: {meta.get('page')}")
            print(doc[:400])  # truncate for readability
        allow_multi = needs_global_context(q["question"])
        context, sources = assemble_chunks(retrieved, allow_multi_section=allow_multi)


        answer = rewrite_with_llm(context, q["question"])
        citations = format_citations(sources)

        print("\n--- Generated Answer ---")
        print(answer)

        print("\n--- Citations ---")
        print(citations)

        print("\n(Expected sections:", q["expected_sections"], ")")
        print("=" * 90)


if __name__ == "__main__":
    
    print("🚀 Running inspection (NO EVAL)...")
    run_inspection()
