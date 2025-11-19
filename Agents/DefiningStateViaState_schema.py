import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from langchain_core.tools import tool


# 定义自定义状态
class CustomState(AgentState) :
    """添加用户偏好到状态"""
    user_preferences: dict

@tool
def get_info() -> str :
    """获取信息"""
    return "Here is some information"

model = ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.1,
    max_tokens=1000
)

agent = create_agent(
    model=model,
    tools=[get_info],
    state_schema=CustomState
)

if __name__ == "__main__" :
    result = agent.invoke({
        "messages" : [{"role" : "user", "content" : "I prefer technical explanations"}],
        "user_preferences" : {"style" : "technical", "verbosity" : "detailed"},
    })

    print("✓ Success!")
    print(f"Response: {result['messages'][-1].content}")
    print(f"User Preferences: {result.get('user_preferences', {})}")