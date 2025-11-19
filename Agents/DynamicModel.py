from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


basic_model = ChatOpenAI(model="gpt-4o-mini")    # 轻量级模型，速度快、成本低
advanced_model = ChatOpenAI(model="gpt-4o")      # 高级模型，能力强、更准确

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10 :
        # 对话较长（超过 10 条消息），使用高级模型处理
        model = advanced_model
    else :
        # 对话较短，使用基础模型即可
        model = basic_model

    request.model = model  # 更新请求中的模型
    return handler(request)  # 继续执行模型调用

agent = create_agent(
    model=basic_model,                            # 默认使用基础模型
    tools=get_weather_for_location,               # 智能体可用的工具列表
    middleware=[dynamic_model_selection]          # 注册动态模型选择中间件
)