import requests
from pathlib import Path
import tempfile
import uuid


def download_file(signed_url: str, filename: str) -> str:
    base_dir = Path(tempfile.gettempdir()) / "rag_ingest"
    base_dir.mkdir(parents=True, exist_ok=True)

    local_path = base_dir / f"{uuid.uuid4()}_{filename}"

    resp = requests.get(signed_url, stream=True, timeout=30)
    resp.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    if not local_path.exists():
        raise RuntimeError("Download failed")

    return str(local_path)
