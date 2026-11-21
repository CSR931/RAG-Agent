import os
from dotenv import load_dotenv

load_dotenv()  # 从 .env 文件加载环境变量

from openai import OpenAI


class LLMClient:
    """统一的 LLM 客户端，支持 DeepSeek / OpenAI / 兼容接口"""

    def __init__(self):
        # 读取 provider，默认为 deepseek
        self.provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
        
        # 调试：打印实际读取到的配置
        print(f"[LLMClient] Provider: {self.provider}")
        #print(f"[LLMClient] Base URL: {os.getenv('LLM_BASE_URL', '未设置')}")
        #print(f"[LLMClient] Model: {os.getenv('LLM_MODEL', '未设置')}")
        #print(f"[LLMClient] API Key: {'已设置' if os.getenv('LLM_API_KEY') else '未设置'}")

        # 如果 provider 是 ollama，强制使用本地配置（防止 .env 残留 DeepSeek 配置）
        if self.provider == "ollama":
            api_key = "ollama"
            base_url = "http://localhost:11434/v1"
            model = os.getenv("LLM_MODEL", "qwen3.5:4b")  # 从 .env 读取模型名，没有则默认
        else:
            # 其他 provider（deepseek/openai）使用 .env 中的值
            api_key = os.getenv("LLM_API_KEY")
            base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
            model = os.getenv("LLM_MODEL", "deepseek-chat")

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def chat(self, messages, temperature=0.1):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content