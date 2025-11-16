import os
# os.environ["OPENAI_API_KEY"] = '你的OpenAI Key'
# from langchain_openai import OpenAI
from langchain_ollama import ChatOllama # Import ChatOllama

llm = model=ChatOllama(
        model="qwen3-vl:8b",
        base_url="http://localhost:11434"  # 默认地址
    )
text = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
print(text)
