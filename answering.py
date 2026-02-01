import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()

OPENROUTER_API_KEY = os.getenv("qwen_key")
def format_citations(sources):
    seen = set()
    citations = []

    for meta in sources:
        key = (meta["section"], meta["page"])
        if key not in seen:
            seen.add(key)
            citations.append(
                f"{meta['section']} (page {meta['page']})"
            )

    return citations


# =========================
# HUGGING FACE (COMMENTED)
# =========================

# HF_TOKEN = os.getenv("token")
#
# API_URL = "https://router.huggingface.co/v1/chat/completions"
# headers = {
#     "Authorization": f"Bearer {HF_TOKEN}",
# }
#
# def rewrite_with_llm(context: str, query: str) -> str:
#     prompt = f"""
# You are a technical writer.
#
# Rewrite the answer using ONLY the information in the context.
# You may paraphrase and rephrase the context, but do not add new facts.
# If the context does not contain the answer, say "Not found in context".
#
# Question:
# {query}
#
# Context:
# {context}
# """
#
#     payload = {
#         "model": "meta-llama/Llama-3.1-8B-Instruct:novita",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         "temperature": 0.0,
#         "max_tokens": 200,
#     }
#
#     response = requests.post(
#         API_URL,
#         headers=headers,
#         json=payload,
#         timeout=60
#     )
#     response.raise_for_status()
#
#     data = response.json()
#     return data["choices"][0]["message"]["content"].strip()


# =========================
# LOCAL OLLAMA (PHI-3 MINI)
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"

def rewrite_with_llm(context: str, query: str) -> str:
    prompt = f"""
You are a careful and knowledgeable assistant answering questions using retrieved document excerpts.

Your goal is to produce a complete, well-structured answer that:
- Covers all relevant aspects implied by the question
- Uses only the provided context
- Does not rely on document structure such as section names or headings
- Does not omit important dimensions (e.g., clinical, technical, economic, social) if they are present in the context

If the context does not contain sufficient information, say so.
If information is distributed across multiple sentences or passages, you should synthesize it into a complete answer.
When a question asks "what are", "why", or "which", present the answer as a concise list if multiple aspects are present in the context.
When a term (e.g., heterogeneity, burden, limitation) can refer to multiple concepts, interpret it in the sense most directly related to the question.

Do not add external knowledge or assumptions.

Question:
{query}

Context:
{context}
"""

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "liquid/lfm-2.5-1.2b-instruct:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.0,
            "max_tokens": 350
        }),
        timeout=300
    )

    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Malformed LLM response: {data}")