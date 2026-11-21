import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.loader import load_texts_from_dir
from src.ingestion.splitter import split_text
from src.retrieval.embedder import Embedder
from src.retrieval.faiss_store import FaissStore
from src.retrieval.bm25 import BM25Retriever
from src.config import load_config


def main():
    cfg = load_config()
    print(f"[配置] embedding 模型: {cfg['embedding_model']}")
    print(f"[配置] chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}")

    # 1. 加载文档（假设返回 [(text, meta), ...]）
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

    # 3. 对每个文档切分，收集所有 chunk（统一为 dict 格式）
    chunks = []  # 每个元素: {"text": str, "meta": dict}
    for item in docs:
        # 兼容 (text, meta) 元组或 {"text":..., "meta":...} 字典
        if isinstance(item, tuple):
            text, meta = item
        elif isinstance(item, dict):
            text = item.get("text", "")
            meta = item.get("meta", {})
        else:
            text = str(item)
            meta = {}

        # 切分当前文档文本
        segs = split_text(text, cfg["chunk_size"], cfg["chunk_overlap"])
        for seg in segs:
            # 如果 split_text 返回的是字符串，则包装成字典
            if isinstance(seg, str):
                chunks.append({"text": seg, "meta": meta})
            else:
                # 如果已经是 dict（包含 text/meta），直接添加
                chunks.append(seg)

    print(f"[切片] 共生成 {len(chunks)} 个文本块")

    # 4. 向量化
    print("[向量化] 正在生成 embedding...")
    texts = [c["text"] for c in chunks]
    vectors = emb.encode(texts)

    # 5. 存入 FAISS
    print("[索引] 正在构建 FAISS 索引...")
    store = FaissStore(dim)
    store.add(vectors, chunks)
    os.makedirs("data/index", exist_ok=True)
    store.save("data/index/idx")
    print(f"✅ FAISS 索引已保存，共 {len(chunks)} 个片段")

    # 6. 构建 BM25 索引
    print("[索引] 正在构建 BM25 索引...")
    bm25 = BM25Retriever()
    bm25.fit(chunks)  # chunks 必须是 list of dict（含 text）
    with open("data/index/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"✅ BM25 索引已保存，共 {len(bm25.docs)} 个片段")


if __name__ == "__main__":
    main()
