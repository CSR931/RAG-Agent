import time
import functools
import json


# ---------- 装饰器 ----------
def tool(name, description):
    """标记一个函数为工具，赋予 _tool_name 和 _tool_desc 属性"""
    def deco(func):
        func._tool_name = name
        func._tool_desc = description
        return func
    return deco


def with_timeout(seconds=10):
    """简单的超时装饰器（同步阻塞版，生产建议用 concurrent.futures）"""
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            if time.time() - start > seconds:
                return "【工具超时】"
            return result
        return wrapper
    return deco


# ---------- 工具函数 ----------
@tool("rag_search", "用知识库检索相关段落。参数: query(str)")
@with_timeout(15)
def rag_search(query: str, store=None, embedder=None, top_k=4):
    """检索工具（实际调用时通过闭包注入 store/embedder）"""
    vec = embedder.encode([query])
    hits = store.search(vec[0], top_k=top_k)
    return "\n---\n".join(
        f"[{i+1}] {h['text']} | 来源: {h['meta'].get('source','未知')}"
        for i, h in enumerate(hits)
    )


@tool("calculator", "计算算术表达式。参数: expression(str)")
@with_timeout(5)
def calculator(expression: str):
    try:
        # 安全 eval，只允许基本运算
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算错误: {e}"


@tool("web_search", "联网搜索。参数: query(str)")
@with_timeout(20)
def web_search(query: str):
    # 暂为桩，后续可接 Tavily / SerpAPI
    return f"[联网搜索占位] 关于「{query}」的结果待接入"


# ---------- 工具注册中心 ----------
TOOL_REGISTRY = {}

def register(tool_func):
    TOOL_REGISTRY[tool_func._tool_name] = tool_func

# 注册所有工具
for f in [rag_search, calculator, web_search]:
    register(f)
