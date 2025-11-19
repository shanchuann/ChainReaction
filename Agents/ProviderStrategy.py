from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

# 如果用 OpenAI/Claude
agent = create_agent(
    model="gpt-4o",  # 或 "claude-3-5-sonnet-20241022"
    response_format=ProviderStrategy(ContactInfo)  # 推荐
)

# 如果用其他模型（如 DeepSeek）
agent = create_agent(
    model="deepseek-chat",
    response_format=ToolStrategy(ContactInfo)  # 必须用 ToolStrategy
)

# LangChain 会自动检测：
# - 如果模型支持 ProviderStrategy → 自动使用
# - 如果模型不支持 → 自动降级到 ToolStrategy

agent = create_agent(
    model="gpt-4o",
    response_format=ContactInfo  # 直接传 schema，LangChain 自动选择最优方案
)