import math
import re
from collections import Counter
from typing import List, Dict, Any


class BM25Retriever:
    """基于分词词频的稀疏召回，与 FAISS 的稠密召回互补"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[Dict[str, Any]] = []
        self.tokens: List[List[str]] = []
        self.df: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.N: int = 0
        self._fitted = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.findall(r"[\u4e00-\u9fa5]|[A-Za-z0-9]+", text.lower()) if t.strip()]

    def fit(self, docs: List[Dict[str, Any]]):
        """docs: [{"text": str, "meta": dict}, ...]"""
        self.docs = docs
        self.tokens = [self._tokenize(d["text"]) for d in docs]
        self.N = len(self.tokens)
        self.avgdl = sum(len(t) for t in self.tokens) / max(self.N, 1)
        for toks in self.tokens:
            for w in set(toks):
                self.df[w] += 1
        # IDF 用 BM25+ 的常见形式
        for w, df in self.df.items():
            self.idf[w] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
        self._fitted = True

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if not self._fitted:
            return []
        q = Counter(self._tokenize(query))
        scores = []
        for i, toks in enumerate(self.tokens):
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for w, cnt in q.items():
                if w not in self.idf:
                    continue
                f = tf[w]
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                s += self.idf[w] * f * (self.k1 + 1) / denom
            scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, s in scores[:top_k]:
            doc = dict(self.docs[i])
            doc["score"] = round(s, 4)
            results.append(doc)
        return results
