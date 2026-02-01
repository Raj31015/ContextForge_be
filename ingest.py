from pdfreader import chunk_multiple_pdfs

from embeddings import index_chunks
from storage import download_file
def ingest_document(doc_id: str, signed_url: str, filename: str):
    print("B: entered ingest_document", flush=True)
    local_path = download_file(signed_url, filename)
    print("C: downloaded file:", local_path, flush=True)
    assert local_path is not None, "download_file returned None"
    assert isinstance(local_path, str), f"local_path is {type(local_path)}"

    chunks = chunk_multiple_pdfs([local_path])
    print("D: chunk_multiple_pdfs produced", len(chunks), "chunks", flush=True)
    assert chunks is not None, "chunk_multiple_pdfs returned None"
    assert len(chunks) > 0, "No chunks produced"

    for c in chunks:
        assert "metadata" in c, "Chunk missing metadata"
        c["metadata"]["doc_id"] = doc_id
        c["metadata"]["source"] = filename
  
    index_chunks(chunks)
