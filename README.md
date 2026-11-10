## 核心逻辑主线

<mark>原始文档 → 加载 → 切片 → 向量化 → 存入FAISS → 检索召回 → Agent推理+调工具 → 生成回答<mark>

• ingestion/ 解决"把资料读进来、切成小块"

• retrieval/ 解决"把这些块变成向量、快速找相关片段"（FAISS + BM25 + 重排三件套）

• agent/ 解决"拿到片段后像人一样思考、调工具、给答案"（ReAct 循环）

• scripts/ 是"一键建索引、一键对话"的入口
### 项目结构
```
ragent-bench/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   └── docs/          # 先放 3 篇小文档
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── splitter.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedder.py
│   │   └── faiss_store.py
│   └── agent/
│       ├── __init__.py
│       ├── llm_client.py
│       ├── tools.py
│       ├── prompts.py
│       └── react_agent.py
└── scripts/
    ├── ingest.py
    └── demo_chat.py
```

### Step1 建立索引如图

<img width="456" height="118" alt="屏幕截图 2026-11-10 212628" src="https://github.com/user-attachments/assets/b20b8cd1-2489-40f3-85ad-d681dce7b612" />
