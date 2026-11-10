import re

def split_text(text: str, chunk_size=400, overlap=80):
    sentences = re.split(r"(?<=[。！？.!?])\s+", text.strip())
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        if cur_len + len(s) > chunk_size and cur:
            chunks.append(" ".join(cur))
            back, bl = [], 0
            for x in reversed(cur):
                back.append(x); bl += len(x)
                if bl >= overlap: break
            cur = list(reversed(back)); cur_len = bl
        cur.append(s); cur_len += len(s)
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]
