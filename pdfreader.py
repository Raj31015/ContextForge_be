import re
import unicodedata
from collections import Counter
from pypdf import PdfReader
from utils import embed_one,cosine_similarity,update_centroid,embed_batch
HEADER_FOOTER_LINES = 3
REPEAT_THRESHOLD = 0.6


def detect_boilerplate_lines(reader: PdfReader):
    header_lines, footer_lines = [], []

    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        header_lines.extend(lines[:HEADER_FOOTER_LINES])
        footer_lines.extend(lines[-HEADER_FOOTER_LINES:])

    counter = Counter(header_lines + footer_lines)
    total_pages = len(reader.pages)

    return {
        line for line, freq in counter.items()
        if freq / total_pages >= REPEAT_THRESHOLD
    }



def normalize_bullet(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)
    return re.sub(r"^[•▪–—◦]+", "", line).strip()


def is_figure_or_table(line: str) -> bool:
    return bool(re.match(r"^(Figure|Table)\s+\d+", line, re.I))


def is_section_heading(line: str):
    m = re.match(r"^(\d+(\.\d+)+)\s*(.+)", line)
    if m:
        return f"{m.group(1)} {m.group(3).strip()}"
    return None


def is_chapter(line: str):
    m = re.match(r"^CHAPTER\s+(\d+)", line, re.I)
    return int(m.group(1)) if m else None


def should_merge(prev, curr):
    if prev.endswith((".", "?", "!")):
        return False
    return curr and curr[0].islower()

def process_page(page, page_number, source, boilerplate, state):
    raw = page.extract_text()
    if not raw:
        return []

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    lines = [l for l in lines if l not in boilerplate]

    blocks = []
    buffer = []
    current_section = state.get("section")

    for line in lines:
        if is_figure_or_table(line):
            continue

        chapter = is_chapter(line)
        if chapter is not None:
            state["chapter"] = chapter
            continue

        section = is_section_heading(line)
        if section:
            if buffer:
                blocks.append({
                    "text": "\n\n".join(buffer),
                    "metadata": {
                        "source": source,
                        "page": page_number,
                        "chapter": state.get("chapter") if state.get("chapter") is not None else -1,
                        "section": current_section if current_section is not None else "unknown"
                                            }
                })
                buffer = []

            current_section = section
            state["section"] = section
            state["found_section"]=True
            continue

        line = normalize_bullet(line)

        if buffer and should_merge(buffer[-1], line):
            buffer[-1] += " " + line
        else:
            buffer.append(line)

    if buffer:
        blocks.append({
            "text": "\n\n".join(buffer),
            "metadata": {
                "source": source,
                "page": page_number,
                "chapter": state.get("chapter"),
                "section": current_section
            }
        })

    return blocks
def paragraph_blocks_from_pages(reader: PdfReader, source: str):
    blocks = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if len(p.strip()) > 40
        ]

        for p in paragraphs:
            blocks.append({
                "text": p,
                "metadata": {
                    "source": source,
                    "page": i + 1,
                    "chapter": -1,
                    "section": "unknown"
                }
            })

    return blocks
def fixed_size_blocks(reader: PdfReader, source: str):
    full_text = []

    for page in reader.pages:
        t = page.extract_text()
        if t:
            full_text.append(t)

    text = "\n".join(full_text)
    words = text.split()

    blocks = []
    chunk_size = 180
    overlap = 30
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        blocks.append({
            "text": chunk,
            "metadata": {
                "source": source,
                "page": -1,
                "chapter": -1,
                "section": "unknown"
            }
        })

        start += chunk_size - overlap

    return blocks


def process_pdf(path: str):
    reader = PdfReader(path)
    boilerplate = detect_boilerplate_lines(reader)

    state = {"chapter": None, "section": None,"found_section":False}
    outputs = []

    for i, page in enumerate(reader.pages):
        if i>5:
            break

        page_blocks = process_page(
            page=page,
            page_number=i + 1,
            source=path,
            boilerplate=boilerplate,
            state=state
        )
        outputs.extend(page_blocks)

    return outputs,state["found_section"]
def chunk_text(
    text: str,
    chunk_size: int = 180,
    overlap: int = 30
):
    words = text.split()
    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text
            })
            chunk_id += 1

        start += chunk_size - overlap

    return chunks
def chunk_section_blocks(section_blocks):
    all_chunks = []
    global_chunk_id=0
    for block in section_blocks:
        section_text = block["text"]
        meta = block["metadata"]

        chunks = chunk_text(section_text)

        for c in chunks:
            chunk_metadata = {
    "source": str(meta.get("source") or ""),
    "page": int(meta["page"]) if isinstance(meta.get("page"), int) else -1,
    "chapter": int(meta["chapter"]) if isinstance(meta.get("chapter"), int) else -1,
    "section": str(meta.get("section") or "unknown"),
    "chunk_id": int(c.get("chunk_id", -1)),
    "global_chunk_id": int(global_chunk_id),
}

       
            all_chunks.append({
                "text": c["text"],
                "metadata": chunk_metadata
            })
            global_chunk_id+=1

    return all_chunks
def semantic_chunk_blocks(
    blocks,
    max_tokens=800,
    min_tokens=200,
    sim_threshold=0.78,
    batch_size=16
):
    chunks = []
    current = []
    current_tokens = 0
    current_emb = None
    current_count = 0
    global_chunk_id = 0

    # 🔹 NEW: embed everything once
    texts = [b["text"] for b in blocks]
    embeddings = embed_batch(texts, batch_size=batch_size)

    for block, emb in zip(blocks, embeddings):
        tokens = len(block["text"].split())

        if not current:
            current = [block]
            current_emb = emb
            current_count = 1
            current_tokens = tokens
            continue

        sim = cosine_similarity(current_emb, emb)

        if (
            sim < sim_threshold and current_tokens >= min_tokens
        ) or current_tokens + tokens > max_tokens:
            chunks.append(build_chunk(current, global_chunk_id))
            global_chunk_id += 1

            current = [block]
            current_emb = emb
            current_count = 1
            current_tokens = tokens
        else:
            current.append(block)
            current_emb = update_centroid(current_emb, emb, current_count)
            current_count += 1
            current_tokens += tokens

    if current:
        chunks.append(build_chunk(current, global_chunk_id))

    return chunks
def build_chunk(blocks, global_chunk_id):
    text = "\n\n".join(b["text"] for b in blocks)
    meta = blocks[0]["metadata"]

    pages = sorted(
        str(p) for p in {b["metadata"].get("page") for b in blocks}
        if isinstance(p, int)
    )

    sections = sorted(
        str(s) for s in {b["metadata"].get("section") for b in blocks}
        if isinstance(s, str)
    )

    return {
        "text": text,
        "metadata": {
            # ✅ SAFE, EXPLICIT METADATA ONLY
            "source": str(meta.get("source") or ""),
            "page": int(meta["page"]) if isinstance(meta.get("page"), int) else -1,
            "chapter": int(meta["chapter"]) if isinstance(meta.get("chapter"), int) else -1,
            "section": str(meta.get("section") or "unknown"),

            "global_chunk_id": int(global_chunk_id),
            "pages": pages,
            "sections": sections,
            "chunk_confidence": round(len(blocks) / 5, 2)
        }
    }


def pdf_to_chunks(pdf_path: str):
    reader = PdfReader(pdf_path)

    section_blocks, has_sections = process_pdf(pdf_path)



    paragraph_blocks = paragraph_blocks_from_pages(reader, pdf_path)



    fixed_blocks = fixed_size_blocks(reader, pdf_path)
    
    blocks = section_blocks if has_sections else paragraph_blocks or fixed_blocks

    return blocks


"""fallback from section based to paragraph chunking or fixed size no regex bs === done
 better preprocessing maybe=
"""
def chunk_multiple_pdfs(pdf_paths):
    all_chunks = []

    for pdf_path in pdf_paths:
        blocks = pdf_to_chunks(pdf_path)
        chunks = semantic_chunk_blocks(blocks)


        for c in chunks:
            c["metadata"]["doc_id"] = pdf_path

        all_chunks.extend(chunks)

    return all_chunks