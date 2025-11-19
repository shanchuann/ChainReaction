import os
from typing import Any
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, AgentMiddleware
from typing_extensions import NotRequired


# 定义自定义状态
class CustomState(AgentState) :
    """扩展 AgentState，添加用户偏好"""
    user_preferences: NotRequired[dict]
    model_call_count: NotRequired[int]


# 定义自定义中间件
class PreferencesMiddleware(AgentMiddleware[CustomState]) :
    """根据用户偏好调整模型行为的中间件"""
    state_schema = CustomState

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None :
        """在模型调用前，根据用户偏好注入提示"""
        preferences = state.get("user_preferences", {})
        style = preferences.get("style", "default")
        verbosity = preferences.get("verbosity", "normal")

        # 根据偏好创建系统提示后缀
        if style == "technical" :
            prompt_suffix = "\n\n[NOTE: User prefers technical explanations with detailed information.]"
        elif style == "simple" :
            prompt_suffix = "\n\n[NOTE: User prefers simple, easy-to-understand explanations.]"
        else :
            prompt_suffix = ""

        if verbosity == "detailed" :
            prompt_suffix += "\n[NOTE: User prefers detailed responses with comprehensive coverage.]"
        elif verbosity == "brief" :
            prompt_suffix += "\n[NOTE: User prefers brief, concise responses.]"

        # 可以在这里修改消息或返回信息用于日志
        print(f"✓ User Preferences Applied: style={style}, verbosity={verbosity}")
        print(f"  Call #{state.get('model_call_count', 0) + 1}")
        return None

    def after_model(self, state: CustomState, runtime) -> dict[str, Any] | None :
        """在模型调用后，更新调用计数"""
        count = state.get("model_call_count", 0)
        return {"model_call_count" : count + 1}


# 简单的示例工具
from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str :
    """获取指定位置的天气信息"""
    return f"The weather in {location} is sunny and 22°C"


@tool
def calculate(expression: str) -> str :
    """执行简单数学计算"""
    try :
        result = eval(expression)
        return str(result)
    except :
        return "Invalid expression"

model = ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.1,
    max_tokens=1000
)

agent = create_agent(
    model=model,
    tools=[get_weather, calculate],
    middleware=[PreferencesMiddleware()]
)

if __name__ == "__main__" :
    # 测试 1: 技术型用户的详细说明
    print("Test 1: Technical user with detailed preferences")
    result1 = agent.invoke({
        "messages" : [
            {
                "role" : "user",
                "content" : "What is machine learning? Also, what's the weather in Beijing?"
            }
        ],
        "user_preferences" : {
            "style" : "technical",
            "verbosity" : "detailed"
        },
        "model_call_count" : 0
    })
    print(f"\n✓ Final Response:\n{result1['messages'][-1].content}\n")

    # 测试 2: 简洁型用户
    print("Test 2: Simple user with brief preferences")
    result2 = agent.invoke({
        "messages" : [
            {
                "role" : "user",
                "content" : "Explain what AI is and calculate 2+2"
            }
        ],
        "user_preferences" : {
            "style" : "simple",
            "verbosity" : "brief"
        },
        "model_call_count" : 0
    })
    print(f"\n✓ Final Response:\n{result2['messages'][-1].content}\n")

    # 测试 3: 默认用户（无偏好）
    print("Test 3: Default user (no preferences)")
    result3 = agent.invoke({
        "messages" : [
            {
                "role" : "user",
                "content" : "What's 10 * 5?"
            }
        ],
        "model_call_count" : 0
    })
    print(f"\n✓ Final Response:\n{result3['messages'][-1].content}\n")