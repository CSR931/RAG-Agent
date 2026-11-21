import os
from .tools import TOOL_REGISTRY


def tool_descriptions():
    """生成工具描述文本，供 system prompt 使用"""
    lines = []
    for name, func in TOOL_REGISTRY.items():
        desc = getattr(func, "_tool_desc", "无描述")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def build_system_prompt(tool_descs: str) -> str:
    max_steps = int(os.getenv("MAX_AGENT_STEPS", 5))
    return f"""你是具备工具调用能力的检索增强智能体。
遇到事实问题先检索知识库；需要计算用计算器；知识库不足可联网。
严格按以下格式输出：

Question: 用户问题
Thought: 你的推理计划
Action: 工具名（只能从下面选）
Action Input: JSON参数
Observation: 工具结果
...（可多轮 Thought/Action/Observation）
Thought: 已有足够信息
Final Answer: 最终答案，并给出引用来源

可用工具：
{tool_descs}

规则：
1. 每次只调用一个工具。
2. Action 必须是工具名，Action Input 必须是合法 JSON。
3. 最多思考 {max_steps} 轮，避免死循环。
4. 答案必须基于 Observation，不得编造。
"""
