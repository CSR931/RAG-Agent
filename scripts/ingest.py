import os
import sys

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.ingestion.loader import load_texts_from_dir
from src.ingestion.splitter import split_text
from src.retrieval.embedder import Embedder
from src.retrieval.faiss_store import FaissStore


def main():
    cfg = load_config()
    print(f"[配置] embedding 模型: {cfg['embedding_model']}")
    print(f"[配置] chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}")

    # 1. 加载文档
    docs = load_texts_from_dir("data/docs")
    if not docs:
        print("⚠️  data/docs/ 下没有找到文档，请先放入 .txt/.md/.pdf 文件")
        return
    print(f"[加载] 共读取 {len(docs)} 个文档")

    # 2. 初始化 embedding 模型
    print("[模型] 正在加载 embedding 模型（首次运行会下载）...")
    emb = Embedder(cfg["embedding_model"])
    dim = emb.get_dim()
    print(f"[模型] 向量维度: {dim}")

    # 3. 切片
    chunks, metas = [], []
    for text, meta in docs:
        for c in split_text(text, cfg["chunk_size"], cfg["chunk_overlap"]):
            chunks.append(c)
            metas.append(meta)
    print(f"[切片] 共生成 {len(chunks)} 个文本块")

    # 4. 向量化
    print("[向量化] 正在生成 embedding...")
    vectors = emb.encode(chunks)

    # 5. 存入 FAISS
    store = FaissStore(dim)
    store.add(vectors, chunks, metas)

    # 6. 保存
    os.makedirs("data/index", exist_ok=True)
    store.save("data/index/idx")
    print(f"✅ 建完索引，共 {len(chunks)} 个片段，已保存到 data/index/idx.faiss")


if __name__ == "__main__":
    main()