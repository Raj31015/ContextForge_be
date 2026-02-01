from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ingest import ingest_document
from query import answer_query

app = FastAPI(title="RAG Engine")

# -------- Request Models --------

class IngestRequest(BaseModel):
    doc_id: str
    signed_url: str   # path in R2 / Supabase Storage
    filename: str

class QueryRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None


# -------- Routes --------

@app.post("/ingest")
def ingest(req: IngestRequest):
    print("A: entered /ingest route", flush=True)
    try:
        ingest_document(
            doc_id=req.doc_id,
            signed_url=req.signed_url,
            filename=req.filename
        )
        return {"status": "indexed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query(req: QueryRequest):
    try:
        result = answer_query(
            question=req.question,
            doc_ids=req.doc_ids
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
