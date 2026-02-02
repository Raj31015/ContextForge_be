# embeddings.py
import json
from dotenv import load_dotenv
from utils import safe_int
from pinecone_client import get_index,pc

load_dotenv()

index_name = get_index()
def get_indexed_ids(index, namespace="ns1"):#no fixed nampespace
    stats = index.describe_index_stats(namespace=namespace)
    count = stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
    return count
def index_chunks(chunks, batch_size=96):
    """
    chunks: list of dicts produced by semantic chunking
    """
    print("E: entered index_chunks with", len(chunks), "chunks", flush=True)
    index = get_index()
    indexed_count = safe_int(get_indexed_ids(index))

    # idempotent guard
    chunks_to_index = [
        c for c in chunks
        if safe_int(c["metadata"].get("global_chunk_id")) >= indexed_count
    ]

    for i in range(0, len(chunks_to_index), batch_size):
        batch = chunks_to_index[i:i + batch_size]
        if not batch:
            continue

        # ---- 1. Texts for embedding ----
        texts = [c["text"] for c in batch]
        if not texts:
            continue

        # ---- 2. Dense embeddings (semantic) ----
        dense_embeddings = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=texts,
            parameters={
                "input_type": "passage",
                "truncate": "END"
            }
        )

        # ---- 3. Sparse embeddings (lexical) ----
        sparse_embeddings = pc.inference.embed(
            model="pinecone-sparse-english-v0",
            inputs=texts,
            parameters={
                "input_type": "passage",
                "truncate": "END"
            }
        )

        # ---- 4. Build records ----
        print("F: building records", flush=True)
        records = []

        for chunk, de, se in zip(batch, dense_embeddings, sparse_embeddings):
            records.append({
                "id": str(chunk["metadata"]["global_chunk_id"]),
                "values": de["values"],
                "sparse_values": {
                    "indices": se["sparse_indices"],
                    "values": se["sparse_values"]
                },
                "metadata": {
                    **chunk["metadata"],
                    "chunk_text": chunk["text"]
                }
            })
        

        index.upsert(
            vectors=records,
            namespace="ns1" #no fixed namespace
        )


