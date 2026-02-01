from pinecone_client import get_index

index = get_index()


# for cid in missing:
#     fetched = index.fetch(ids=[cid], namespace="ns1")
#     vec = fetched["vectors"][cid]

#     text = vec["metadata"]["chunk_text"]

#     try:
#         role = predict_discourse_role(text)
#     except Exception:
#         role = "other"

#     index.upsert(
#         namespace="ns1",
#         vectors=[{
#             "id": cid,
#             "values": vec["values"],  # reuse existing vector
#             "metadata": {
#                 **vec["metadata"],
#                 "role": role
#             }
#         }]
#     )

def copy_namespace(src_ns, dst_ns, batch_size=100):
    # Get all vector IDs from src namespace
    stats = index.describe_index_stats(namespace=src_ns)
    total = stats["namespaces"][src_ns]["vector_count"]

    print(f"📦 Copying {total} vectors from {src_ns} → {dst_ns}")

    # We must query in batches because fetch needs IDs
    offset = 0
    copied = 0

    while copied < total:
        res = index.query(
            namespace=src_ns,
            top_k=batch_size,
            vector=[0.0] * 1024,
            include_values=True,
            include_metadata=True
        )

        if not res["matches"]:
            break

        vectors = []
        for m in res["matches"]:
            vectors.append({
                "id": m["id"],
                "values": m["values"],
                "metadata": m["metadata"]
            })

        index.upsert(
            namespace=dst_ns,
            vectors=vectors
        )

        copied += len(vectors)
        print(f"✅ Copied {copied}/{total}")



index.delete(
    delete_all=True,
    namespace="ns2"
    
)

index.delete(
    delete_all=True,
    namespace="ns1"
    
)
print(index.describe_index_stats())
