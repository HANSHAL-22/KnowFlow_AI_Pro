import io
import re
import fitz


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_words(text: str, chunk_size: int, overlap: int):
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    out = []
    start = 0
    chunk_no = 1
    while start < len(words):
        end = min(len(words), start + chunk_size)
        out.append((chunk_no, " ".join(words[start:end])))
        chunk_no += 1
        if end == len(words):
            break
        start += step
    return out


def ingest_pdf(pdf_bytes: bytes, source_name: str, chunk_size=650, overlap=110):
    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    chunks = []
    pages_with_text = 0

    for pno, page in enumerate(doc, 1):
        text = clean_text(page.get_text("text"))
        if not text:
            continue
        pages_with_text += 1
        for cno, chunk in split_words(text, chunk_size, overlap):
            chunks.append({
                "text": chunk,
                "source": source_name,
                "page": pno,
                "chunk_id": f"{source_name}|p{pno}|c{cno}",
            })

    total_pages = len(doc)
    doc.close()
    return chunks, {"pages": total_pages, "pages_with_text": pages_with_text}
