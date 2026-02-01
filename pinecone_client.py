import os
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv
load_dotenv()
INDEX_NAME = "contextforge"

pc = Pinecone(api_key=os.getenv("pc_key"))

def ensure_index():
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,              # llama-text-embed-v2 dim
            metric="dotproduct",         # REQUIRED for sparse + dense
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
def get_index():
    ensure_index()
    return pc.Index(INDEX_NAME)
