from retrieval import hybrid_search,needs_global_context,assemble_chunks,is_answerable
from answering import rewrite_with_llm,format_citations
def is_answerable(results, min_abs_score=0.3):
    """
    Checks whether the top retrieved chunk is sufficiently relevant
    to attempt an answer.
    """
    if not results:
        return False

    top_score = results[0][0]
    return top_score >= min_abs_score

def answer_query(
    question: str,
    doc_ids: list[str] | None = None,
):
    results = hybrid_search(
        query=question,
        doc_ids=doc_ids
    )
    if not is_answerable(results):
        return {
            "answer": "The uploaded documents do not contain enough information to answer this question.",
            "citations": [],
            "context": "",
        }

    allow_multi = needs_global_context(question)

    context, sources = assemble_chunks(
        results,
        allow_multi_section=allow_multi
    )

    if not context:
        return {
            "answer": "The retrieved documents do not contain enough information to answer this question.",
            "citations": [],
        }

    answer = rewrite_with_llm(context, question)
    citations = format_citations(sources)

    return {
        "answer": answer,
        "citations": citations,
        "context":context
    }
