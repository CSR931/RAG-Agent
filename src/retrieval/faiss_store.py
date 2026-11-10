import faiss
import numpy as np
import pickle
from typing import List, Dict, Optional


class FaissStore:
    """
    FAISS 向量存储，使用内积（余弦相似度）检索，要求向量已归一化。
    支持添加、搜索、保存、加载。
    """

    def __init__(self, dim: int):
        """
        初始化 FAISS 索引（IndexFlatIP）
        :param dim: 向量维度
        """
        self.index = faiss.IndexFlatIP(dim)   # 余弦相似度，输入需归一化
        self.texts: List[str] = []            # 存储原始文本
        self.metas: List[Dict] = []           # 存储元数据（如来源、页码）

    def add(self, vectors: np.ndarray, texts: List[str], metas: Optional[List[Dict]] = None):
        """
        批量添加向量和对应的文本/元数据
        :param vectors: shape (n, dim)，float32，应已归一化
        :param texts: 文本列表
        :param metas: 元数据列表，若为 None 则填充空字典
        """
        assert len(vectors) == len(texts), "vectors 和 texts 长度不一致"
        if metas is None:
            metas = [{} for _ in range(len(texts))]
        self.index.add(vectors.astype(np.float32))
        self.texts.extend(texts)
        self.metas.extend(metas)

    def search(self, vec: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        检索最相似的 top_k 个结果
        :param vec: 查询向量，shape (dim,) 或 (1, dim)，float32，已归一化
        :param top_k: 返回数量
        :return: 列表，每个元素包含 text, meta, score
        """
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        vec = vec.astype(np.float32)
        scores, indices = self.index.search(vec, top_k)
        results = []
        for s, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS 返回 -1 表示无效索引
                continue
            results.append({
                "text": self.texts[idx],
                "meta": self.metas[idx],
                "score": float(s)
            })
        return results

    def save(self, path_prefix: str):
        """
        保存索引和元数据到磁盘
        :param path_prefix: 前缀，例如 "data/index/idx"，将生成 .faiss 和 .meta.pkl 两个文件
        """
        faiss.write_index(self.index, f"{path_prefix}.faiss")
        with open(f"{path_prefix}.meta.pkl", "wb") as f:
            pickle.dump({"texts": self.texts, "metas": self.metas}, f)

    @classmethod
    def load(cls, path_prefix: str, dim: int) -> "FaissStore":
        """
        从磁盘加载索引和元数据
        :param path_prefix: 与 save 时一致的前缀
        :param dim: 向量维度，用于构造实例
        :return: FaissStore 实例
        """
        store = cls(dim)
        store.index = faiss.read_index(f"{path_prefix}.faiss")
        with open(f"{path_prefix}.meta.pkl", "rb") as f:
            data = pickle.load(f)
            store.texts = data["texts"]
            store.metas = data["metas"]
        return store

    def __len__(self) -> int:
        return len(self.texts)
