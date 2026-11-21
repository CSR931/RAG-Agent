import json
import re
import os
from .llm_client import LLMClient
from .tools import TOOL_REGISTRY
from .prompts import build_system_prompt, tool_descriptions


class ReActAgent:
    def __init__(self, llm=None, rag_fn=None):
        self.llm = llm or LLMClient()
        self.rag_fn = rag_fn  # 检索函数，接收 query 返回 context 字符串
        self.max_steps = int(os.getenv("MAX_AGENT_STEPS", 5))
        # 构建 system prompt（这里使用了 prompt 模板）
        self.sys_prompt = build_system_prompt(tool_descriptions())
        self.store = None
        self.bm25 = None
        self.embedder = None
        self.fusion = None
        self.reranker = None

    def _bind_rag(self, store, bm25, embedder, fusion=None, reranker=None):
        """
        绑定检索组件
        :param store: FaissStore 对象
        :param bm25: BM25Retriever 对象
        :param embedder: Embedder 对象
        :param fusion: RRFusion 对象（可选）
        :param reranker: CrossEncoderReranker 对象（可选）
        """
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.fusion = fusion
        self.reranker = reranker
        
        def fn(query):
            dense = store.search(embedder.encode([query])[0], top_k=20)
            sparse = bm25.search(query, top_k=20)
            hits = fusion.fuse([dense, sparse], top_n=10) if fusion else dense
            if reranker:
                hits = reranker.rerank(query, hits, top_n=4)
            else:
                hits = hits[:4]
            return "\n---\n".join(
                f"[{i+1}] {h['text']}" for i, h in enumerate(hits)
            )
        self.rag_fn = fn


    def _parse_action(self, text):
        """解析模型输出中的 Action、Action Input、Final Answer"""
        # 匹配 Action: xxx
        action_m = re.search(r"Action:\s*(\w+)", text)
        # 匹配 Action Input: {...} 或 Action Input: "xxx"
        input_m = re.search(r"Action Input:\s*(\{.*?\}|[\"'].*?[\"'])", text, re.DOTALL)
        # 匹配 Final Answer: ...
        final_m = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)

        action = action_m.group(1).strip() if action_m else None
        action_input = {}
        if input_m:
            raw = input_m.group(1).strip()
            try:
                action_input = json.loads(raw)
            except json.JSONDecodeError:
                # 如果不是 JSON，尝试当作普通字符串参数
                action_input = {"query": raw.strip('"\'')}
        final = final_m.group(1).strip() if final_m else None
        return action, action_input, final

    def run(self, question: str):
        """执行 ReAct 循环"""
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ]
        trace = []  # 记录思考轨迹

        for step in range(self.max_steps):
            output = self.llm.chat(messages)
            messages.append({"role": "assistant", "content": output})
            trace.append(output)

            action, action_input, final = self._parse_action(output)
            if final:
                return {"answer": final, "trace": trace}

            if action and action in TOOL_REGISTRY:
                # 如果是 rag_search，使用绑定的检索函数
                if action == "rag_search" and self.rag_fn:
                    query = action_input.get("query", "")
                    obs = self.rag_fn(query)
                else:
                    # 其他工具直接调用
                    try:
                        obs = TOOL_REGISTRY[action](**action_input)
                    except Exception as e:
                        obs = f"工具调用出错: {e}"
                messages.append({"role": "user", "content": f"Observation: {obs}"})
            else:
                # 没有解析到有效动作，可能是模型直接给出了答案
                # 尝试将整个输出作为答案返回
                return {"answer": output, "trace": trace}

        # 达到最大步数仍未给出 Final Answer
        return {"answer": "未能在规定步数内完成推理", "trace": trace}
