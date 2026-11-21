from typing import List, Dict


class RRFusion:
    """倒数排名融合，把多路召回合并成一个有序列表"""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, lists: List[List[Dict]], top_n: int = 10) -> List[Dict]:
        scores: Dict[str, float] = {}
        payloads: Dict[str, Dict] = {}
        for hits in lists:
            for rank, doc in enumerate(hits):
                # [修改] 安全获取 id 或 text，确保 key 是字符串
                doc_id = doc.get("id")
                if doc_id:
                    key = str(doc_id)
                else:
                    text = doc.get("text", "")
                    # 如果 text 不是字符串，强制转字符串
                    if not isinstance(text, str):
                        text = str(text)
                    key = text[:80]  # 取前80字符作为key
                payloads[key] = doc
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.k + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out = []
        for key, sc in ranked[:top_n]:
            d = dict(payloads[key])
            d["rrf_score"] = round(sc, 4)
            out.append(d)
        return out
