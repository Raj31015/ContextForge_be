from pinecone_client import pc
import numpy as np
def pinecone_embed(texts):
    """
    texts: List[str]
    returns: List[List[float]]
    """
    res = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=texts,
        parameters={
            "input_type": "passage",
            "truncate": "END"
        }
    )
    return [r["values"] for r in res]
def embed_one(text):
    return pinecone_embed([text])[0]


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def update_centroid(centroid, new_emb, n):
    return [
        (centroid[i] * n + new_emb[i]) / (n + 1)
        for i in range(len(centroid))
    ]

def embed_batch(texts, batch_size=16):
    """
    texts: List[str]
    returns: List[List[float]]
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        res = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=batch,
            parameters={
                "input_type": "passage",
                "truncate": "END"
            }
        )
        all_embeddings.extend([r["values"] for r in res])

    return all_embeddings
def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
