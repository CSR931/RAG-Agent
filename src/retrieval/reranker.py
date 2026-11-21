import os
from typing import List, Dict, Optional


class CrossEncoderReranker:
    """Cross-encoder 重排器，对召回结果进行精排。"""

    def __init__(self, model_name: Optional[str] = None):
        self.model = None
        if model_name:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(model_name)
                print(f"[Reranker] 加载模型: {model_name}")
            except Exception as e:
                print(f"[Reranker] 加载失败: {e}，将跳过重排")

    def rerank(self, query: str, candidates: List[Dict], top_n: int = 4) -> List[Dict]:
        """对候选列表进行重排，返回前 top_n 个。"""
        if not self.model or not candidates:
            return candidates[:top_n]

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        # 按分数降序排列
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_n]]
