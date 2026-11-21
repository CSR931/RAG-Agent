## 核心逻辑主线

<mark>原始文档 → 加载 → 切片 → 向量化 → 存入FAISS → 检索召回 → Agent推理+调工具 → 生成回答<mark>

• ingestion/ 解决"把资料读进来、切成小块"

• retrieval/ 解决"把这些块变成向量、快速找相关片段"（FAISS + BM25 + 重排三件套）

• agent/ 解决"拿到片段后像人一样思考、调工具、给答案"（ReAct 循环）

• scripts/ 是"一键建索引、一键对话"的入口

## 理论学习
**RAG流程**
```
文档 → [分片] → 文本块 → [索引] → 向量库 + BM25索引
                          ↓
用户问题 → [召回] → 混合检索（FAISS + BM25）→ Top-N
                          ↓
                   [重排] → Cross-encoder 精排 → Top-K
                          ↓
                   [生成] → LLM + 上下文 → 带引用的答案
```
**分片**：长文档按不同策略切分为小段文本块

关键参数：

• chunk_size：每段最大长度（字符/token）。

• chunk_overlap：相邻段重叠部分（通常 10%~20%），避免关键信息恰好落在切缝处丢失。<mark>防止上下文截断导致语义断裂<mark>

**索引**：将文本块转换成可快速检索结构，向量索引（稠密检索）和倒排索引（稀疏检索）

稠密检索：用embedding模型把文本转为向量，检索时计算向量之间余弦相似度（内积）。<mark>擅长语义相似<mark>

吸收索引：基于词频统计的传统方法不依赖神经网络，对专有名词、缩写、代码片段特别有效。<mark>擅长精确匹配<mark>

**召回**：给定问题找到最相关K个文本块
1. 将用户问题也用相同的 embedding 模型转为向量。

2. 在向量数据库中搜索，得到 Top-N（如 10 个）。（可选）用BM25搜索得到Top-M

3. 用RRF融合两个结果列表得到最终候选集Top-K

<mark>评估指标：<mark>

• Recall@K：正确答案在前 K 个结果中的比例。

• MRR：第一个正确答案的排名的倒数均值。

• nDCG：考虑排序位置和相关性等级。

如何提高召回率？

• 多路召回（向量 + BM25 + 其他特征）。

• 查询改写（HyDE：先生成假设答案，再用答案向量去检索）。

• 分块策略优化（overlap、语义切分）。

**重排**：用更精细的模型对“问题-候选段”逐对打分，把真正相关的排到最前面

重排可以大幅提升最终答案质量，尤其当召回结果中有多个相似片段时。

<mark>可以代替召回吗？<mark>重排模型慢，不能直接在全量库上跑，必须先用轻量召回缩小范围。

**生成**：将用户问题和重排后的相关文本块一起喂给大语言模型，让其基于证据生成答案。

Promt模板————规定了大模型该怎么说话、用什么格式输出、能调用哪些工具
```
请根据以下资料回答问题。如果资料不足以回答，请如实说不知道。
资料：
{context}
问题：{question}
答案：
```
关键设计：

• 引用来源：要求模型在答案中标明出自哪个段落，增加可信度（如 [1] [2]）。

• 拒绝回答：当 context 不包含答案时，模型应回答“资料中未提及”，而非瞎编。

• 多轮对话：结合历史消息，支持追问。
## 具体实现
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
│   │   └── splitter.py        # 分片
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedder.py        # 索引
│   │   └── faiss_store.py     # 索引+召回
│   └── agent/
│       ├── __init__.py
│       ├── llm_client.py
│       ├── tools.py
│       ├── prompts.py         # 生成
│       └── react_agent.py     # 生成
└── scripts/
    ├── ingest.py
    └── demo_chat.py
```
### 详细说明
一、分片（Chunking）

项目实现：采用句子级递归分割，优先保句子完整，同时用字符级重叠防止切缝丢失信息。

• 策略：按中文/英文标点（。！？.!?）先切句子，然后累积句子直到超过 chunk_size，切出一块；同时用 overlap 字符数从上一块末尾回退，作为下一块开头，保证边界不丢失关键信息。

• 代码关键行：
```
函数 split_text(text, chunk_size=400, overlap=80)
sentences = re.split(r"(?<=[。！？.!?])\s+", text.strip())
然后循环判断 cur_len + len(s) > chunk_size 时切块，并用 back 列表收集最后若干句子直到凑够 overlap 字符。
```

二、索引（Indexing）

用 FAISS 做语义检索，预留 BM25 做关键词精确匹配，后续通过 RRF 融合提升召回鲁棒性。项目同时构建稠密索引（FAISS） 和稀疏索引（BM25），但当前最小版本只做了 FAISS，BM25 放在 bm25.py 中待集成。

2.1 稠密索引（向量化 + FAISS）

• 文件：src/retrieval/embedder.py 和 src/retrieval/faiss_store.py

• 流程：

  1. Embedder.encode(texts) 调用 sentence-transformers 的 all-MiniLM-L6-v2 模型，将文本列表转为归一化的 numpy 数组（shape [N, 384]）。

  2. FaissStore.add(vectors, texts, metas) 将向量加入 IndexFlatIP（内积索引，因向量已归一化，等价于余弦相似度）。

  3. 保存时写入 data/index/idx.faiss 和 idx.meta.pkl。

关键代码：
```
# embedder.py
self.model.encode(texts, normalize_embeddings=True, ...)

# faiss_store.py
self.index = faiss.IndexFlatIP(dim)
self.index.add(vectors.astype(np.float32))
```

2.2 稀疏索引（BM25，可选）

• 文件：src/retrieval/bm25.py

• 实现：用 rank_bm25.BM25Okapi，对分词后的语料建立索引，get_scores 返回分数。

三、召回（Retrieval）

项目实现：

• 文件：src/retrieval/faiss_store.py 的 search 方法，以及未来 hybrid.py 的 rrf_merge。

• 当前版本仅用 FAISS 召回：
```
def search(self, vec, top_k=10):
    scores, indices = self.index.search(vec, top_k)
    # 组装结果列表
```
• 查询向量由 embedder.encode([query]) 得到，然后调用 store.search 获取 Top-10（由环境变量 TOP_K 控制）。

• 后续计划：同时调用 BM25，然后用 rrf_merge 融合两个列表。

面试可说：我们实现了混合检索的第一步——向量召回，后续将加入 BM25 和 RRF 融合，以覆盖语义和精确匹配两种需求。

四、重排（Reranking）

项目实现：

• 文件：src/retrieval/reranker.py，类 CrossEncoderReranker

• 使用 sentence-transformers.CrossEncoder 加载 BAAI/bge-reranker-v2-m3（或通过环境变量指定）。

• 方法 rerank(query, candidates, top_n=4)：

  1. 构造 (query, candidate_text) 对列表。

  2. 调用 model.predict(pairs) 得到分数。

  3. 按分数降序排列，取前 top_n 个。

• 当前最小版本中，reranker 初始化为 None，跳过重排；后续可开启。

面试可说：我们预留了 Cross-encoder 重排模块，可以在召回结果上做二次精排，将最相关的 3-4 个片段送入生成，提升答案质量。

五、生成（Generation）

实现了 ReAct 循环，让模型能自主决定何时检索、何时计算、何时回答。检索结果被当作 observation 注入，LLM 基于证据生成最终答案，避免了纯生成带来的幻觉。

项目实现：

• 文件：src/agent/react_agent.py 中的 ReActAgent.run() 方法，以及 prompts.py 中的 system prompt。

• 流程：

  1. 构建 system prompt，包含工具描述和 ReAct 格式要求。

  2. 将用户问题作为 user message 加入。

  3. 循环调用 LLMClient.chat()，每次返回 assistant 输出。

  4. 解析输出中的 Action 和 Action Input，执行对应工具（如 rag_search 调用检索），将结果作为 Observation 回填。

  5. 直到模型输出 Final Answer 或达到最大步数。

• 检索工具 rag_search 内部调用 self.rag_fn(query)，该函数绑定了 embedder.encode + store.search + 可选重排，返回拼接后的文本块。

• 生成时的上下文就是这些文本块，LLM 据此作答。

关键代码：
```
# react_agent.py
output = self.llm.chat(messages)
action, action_input, final = self._parse_action(output)
if action == "rag_search" and self.rag_fn:
    obs = self.rag_fn(action_input["query"])
messages.append({"role": "user", "content": f"Observation: {obs}"})
```
整体流程图（代码对应）
```
用户问题
    │
    ▼
ReActAgent.run()
    │
    ├─ LLM 生成 Thought + Action
    │      │
    │      ├─ Action = rag_search
    │      │      │
    │      │      ▼
    │      │  embedder.encode(query) → store.search() → [可选 reranker.rerank()] → context
    │      │      │
    │      │      ▼
    │      │  Observation = context
    │      │
    │      └─ Action = calculator / web_search（直接调用对应函数）
    │
    ├─ 重复直到 Final Answer
    │
    ▼
最终答案（带引用）
```

### Step1 建立索引如图

<img width="456" height="118" alt="屏幕截图 2026-11-10 212628" src="https://github.com/user-attachments/assets/b20b8cd1-2489-40f3-85ad-d681dce7b612" />
