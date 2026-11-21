import sys
import os

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.embedder import Embedder
from src.retrieval.faiss_store import FaissStore
from src.retrieval.reranker import CrossEncoderReranker
from src.agent.react_agent import ReActAgent
from src.retrieval.bm25 import BM25Retriever       # [修改] 新增导入
from src.retrieval.hybrid import RRFusion           # [修改] 新增导入
from dotenv import load_dotenv
import pickle

load_dotenv()


def main():
    # 1. 加载 embedding 模型
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"[加载] embedding 模型: {model_name}")
    embedder = Embedder(model_name)
    dim = embedder.get_dim()

    # 2. 加载 FAISS 索引
    index_path = "data/index/idx"
    if not os.path.exists(f"{index_path}.faiss"):
        print("❌ 索引文件不存在，请先运行 scripts/ingest.py")
        return
    store = FaissStore.load(index_path, dim)
    print(f"[加载] FAISS 索引，共 {len(store)} 个片段")

    # [修改] 3. 加载 BM25 索引
    bm25_path = "data/index/bm25.pkl"
    if not os.path.exists(bm25_path):
        print("❌ BM25 索引文件不存在，请先运行 scripts/ingest.py")
        return
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    print(f"[加载] BM25 索引，共 {len(bm25.docs)} 个片段")

    # [修改] 4. 初始化 RRF 融合器
    fusion = RRFusion(k=60)  # k 值可按需调整
    print("[加载] RRF 融合器")

    # 5. 初始化重排器（可选）
    reranker_model = os.getenv("RERANKER_MODEL")
    reranker = CrossEncoderReranker(reranker_model) if reranker_model else None
    if reranker:
        print(f"[加载] 重排器: {reranker_model}")

    # [修改] 6. 创建 Agent 并绑定检索（注意参数顺序：store, bm25, embedder, fusion, reranker）
    agent = ReActAgent()
    agent._bind_rag(store, bm25, embedder, fusion=fusion, reranker=reranker)
    print("✅ Agent 初始化完毕，请输入问题（输入 quit 退出）\n")

    # 7. 交互循环
    while True:
        question = input(">>> ")
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question.strip():
            continue
        result = agent.run(question)
        print("\n🤖 答案:", result["answer"])
        print("\n--- 思考轨迹 ---")
        for i, t in enumerate(result["trace"], 1):
            # 打印完整思考轨迹（不再截断）
            print(f"\n--- Step {i} ---\n{t}\n")
        print("\n" + "=" * 40 + "\n")


if __name__ == "__main__":
    main()
