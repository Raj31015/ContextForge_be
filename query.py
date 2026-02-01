from retrieval import hybrid_search,needs_global_context,assemble_chunks
from answering import rewrite_with_llm,format_citations
def answer_query(
    question: str,
    doc_ids: list[str] | None = None,
):
    results = hybrid_search(
        query=question,
        doc_ids=doc_ids
    )

    allow_multi = needs_global_context(question)

    context, sources = assemble_chunks(
        results,
        allow_multi_section=allow_multi
    )

    if not context:
        return {
            "answer": "The retrieved documents do not contain enough information to answer this question.",
            "sources": [],
        }

    answer = rewrite_with_llm(context, question)
    citations = format_citations(sources)

    return {
        "answer": answer,
        "sources": citations,
    }
