from pathlib import Path


def load_texts_from_dir(doc_dir: str):
    """
    支持 .txt/.md/.pdf，返回 [(text, meta)] 列表
    """
    docs = []
    for p in Path(doc_dir).rglob("*"):
        if p.suffix.lower() in [".txt", ".md"]:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append((text, {"source": str(p)}))
        elif p.suffix.lower() == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(p))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                docs.append((text, {"source": str(p), "pages": len(reader.pages)}))
            except Exception as e:
                print(f"[PDF 读取失败] {p}: {e}")
    return docs
