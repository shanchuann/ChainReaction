# Agents 代理

代理将语言模型与[工具](https://docs.langchain.com/oss/python/langchain/tools)结合，创建能够推理任务、决定使用哪些工具，并迭代地工作以寻求解决方案的系统。[`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)提供了一种生产就绪的代理实现。[LLM 代理在循环中运行工具以实现目标](https://simonwillison.net/2025/Sep/18/agents/)。代理会一直运行，直到满足停止条件——即模型发出最终输出或达到迭代限制。

![image-20251119145224242](https://s2.loli.net/2025/11/19/m1TdYy9rSKC8hJc.png)

## 核心组件

### Model 模型

[模型](https://docs.langchain.com/oss/python/langchain/models)是您代理的推理引擎。它可以通过多种方式指定，支持静态和动态模型选择。

#### Static model 静态模型

静态模型在创建代理时配置一次，并在整个执行过程中保持不变。这是最常见和直接的方法。

从模型标识符字符串初始化静态模型：

```python
from langchain.agents import create_agent

agent = create_agent(
    "deepseek",
    tools=tools
)
```

若需对模型配置进行更多控制，可直接使用提供方包初始化模型实例。在此示例中，我们使用 [`ChatOpenAI`](https://reference.langchain.com/python/integrations/langchain_openai/ChatOpenAI)。参见[聊天模型](https://docs.langchain.com/oss/python/integrations/chat)了解其他可用的聊天模型类。

```python
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
    # ... (other params)
)
```
模型实例让您可以完全控制配置。当您需要设置特定的[参数](https://docs.langchain.com/oss/python/langchain/models#parameters)（如`temperature`、`max_tokens`、`timeouts`、`base_url`和其他提供者特定设置）时，请使用它们。参考[文档](https://docs.langchain.com/oss/python/integrations/providers/all_providers)查看您的模型上可用的参数和方法。

#### Dynamic model 动态模型

动态模型是在运行时根据当前状态和上下文选择的。这使得复杂的路由逻辑和成本优化成为可能。要使用动态模型，请使用 [`@wrap_model_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_model_call) 装饰器创建中间件，该装饰器会修改请求中的模型：

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])
@wrap_model_call def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse: """根据对话复杂度选择模型。""" message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model
如果 message_count > 10: # 使用高级模型处理更长的对话 model = advanced_model
else: model = basic_model

    request.model = model
    return handler(request)

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[dynamic_model_selection]
)
```
```
用户输入 
  ↓
Agent 调用模型
  ↓
[中间件拦截] 检查消息数量
  ↓
消息数 > 10？
  ├─ 是 → 切换到 gpt-4o（高级）
  └─ 否 → 保持 gpt-4o-mini（基础）
  ↓
执行模型调用 → 返回结果
```

> 预绑定模型（已调用 [`bind_tools`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.bind_tools) 的模型）在使用结构化输出时不被支持。如果你需要在结构化输出时进行动态模型选择，请确保传递给中间件的模型没有预绑定。

### Tools  工具

工具使代理能够执行操作。代理通过以下方式超越了简单的模型仅工具绑定：

* 顺序调用多个工具（由单个提示触发）
* 在适当情况下并行调用工具 
* 基于先前结果动态选择工具 
* 工具重试逻辑和错误处理 
* 跨工具调用的状态持久化

更多信息，请参阅[工具](https://docs.langchain.com/oss/python/langchain/tools)。

#### Defining tools 定义工具

将一组工具传递给代理。

> 工具可以指定为普通的 Python 函数或协程。[工具装饰器](https://docs.langchain.com/oss/python/langchain/tools#create-tools)可用于自定义工具名称、描述、参数模式和其他属性。

```python
import os
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

model = ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
    # ... (other params)
)

agent = create_agent(model, tools=[search, get_weather])
```

如果提供的工具列表为空，代理将仅由一个 LLM 节点组成，且不具备调用工具的功能。

#### Tool error handling 工具错误处理

要自定义工具错误的处理方式，请使用 [`@wrap_tool_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_tool_call) 装饰器来创建中间件：

```python
import os
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

model = ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.1,
    max_tokens=100
)

agent = create_agent(
    model=model,
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)
```

当工具失败时，代理将返回一个包含自定义错误消息的[`ToolMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.ToolMessage)：

```python
[
    ...
    ToolMessage(
        content="Tool error: Please check your input and try again. (division by zero)",
        tool_call_id="..."
    ),
    ...
]
```
> **DeepSeek 模型很聪明，它在调用工具之前就识别出了问题，所以工具根本没有被执行，因此中间件也没有捕获到异常。**所以看不到想要的输出是正常的,你可以采用强制调用或者**直接测试中间件**，而不是依赖模型决策

#### Tool use in the ReAct loop ReAct 循环中的工具使用

代理遵循 ReAct（"推理+行动"）模式，在简短的推理步骤和有针对性的工具调用之间交替，并将生成的观察结果输入后续决策中，直到能够提供最终答案。

*Example of ReAct loop ReAct 循环示例*

提示：识别当前最受欢迎的无线耳机并验证其可用性。

  ```
  ================================ Human Message =================================
  Find the most popular wireless headphones right now and check if they're in stock
  ```
 * **推理**： "受欢迎程度具有时效性，我需要使用提供的搜索工具。" 
 * **行动**： 调用 `search_products("wireless headphones")`

  ```
  ================================== Ai Message ==================================
  Tool Calls:
    search_products (call_abc123)
   Call ID: call_abc123
    Args:
      query: wireless headphones
  ```
  ```
  ================================= Tool Message =================================
  Found 5 products matching "wireless headphones". Top 5 results: WH-1000XM5, ...
  ```
  * **推理**：在回答之前，我需要确认排名第一的商品的库存情况。 
  * **行动**：调用 `check_inventory("WH-1000XM5")`

  ```
  ================================== Ai Message ==================================
  Tool Calls:
    check_inventory (call_def456)
   Call ID: call_def456
    Args:
      product_id: WH-1000XM5
  ```
  ```
  ================================= Tool Message =================================
  Product WH-1000XM5: 10 units in stock
  ```
  * **推理**： "我拥有最受欢迎的型号及其库存状态。现在我能够回答用户的问题。" 
  * **行动**：生成最终答案

  ```
  ================================== Ai Message ==================================
  I found wireless headphones (model WH-1000XM5) with 10 units in stock...
  ```
> 想了解更多关于工具的信息，请查看[Tools](https://docs.langchain.com/oss/python/langchain/tools)。

### System prompt 系统提示

您可以通过提供提示来塑造您的代理如何处理任务。[`system_prompt`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent\(system_prompt\))参数可以作为字符串提供：

```python
import os
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

model = ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.1,
    max_tokens=100
)

agent = create_agent(
    model,
    tools=[search],
    system_prompt="You are a helpful assistant. Be concise and accurate."
)
```
当没有提供[`system_prompt`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent\(system_prompt\))时，助手将直接从消息中推断其任务。

#### Dynamic system prompt 动态系统提示
对于需要根据运行时上下文或助手状态修改系统提示的高级用例，你可以使用 [middleware](https://docs.langchain.com/oss/python/langchain/middleware)。[`@dynamic_prompt`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.dynamic_prompt) 装饰器创建了一个中间件，该中间件根据模型请求动态生成系统提示：

```python
import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.tools import tool

class Context(TypedDict):
    user_role: str

model = ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.1,
    max_tokens=100
)

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

agent = create_agent(
    model="gpt-4o",
    tools=[search],
    middleware=[user_role_prompt],
    context_schema=Context
)

# The system prompt will be set dynamically based on context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"}
)
```
> 有关消息类型和格式的更多详细信息，请参阅[消息](https://docs.langchain.com/oss/python/langchain/messages)。有关全面的中间件文档，请参阅[中间件](https://docs.langchain.com/oss/python/langchain/middleware)。

## Invocation 调用

你可以通过向其  [`State`](https://docs.langchain.com/oss/python/langgraph/graph-api#state) 发送更新来调用一个代理。所有代理在其状态中都包含一系列消息；要调用代理，请传递一条新消息：

```python  theme={null}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
```
从代理中流式传输步骤和/或标记，请参阅[流式传输](https://docs.langchain.com/oss/python/langchain/streaming)指南。否则，代理遵循  [LangGraph API](/oss/python/langgraph/use-graph-api)并支持所有相关方法，例如`stream`和`invoke`。

## Advanced concepts 高级概念

### Structured output 结构化输出

在某些情况下，您可能希望代理以特定的格式返回输出。LangChain 通过 `response_format` 参数提供了结构化输出的策略。

### ToolStrategy 工具策略

`ToolStrategy` 使用人工工具调用生成结构化输出。这适用于任何支持工具调用的模型：

```pythonwrap theme={null}
import os
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

# Create DeepSeek model with proper configuration
model = ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.1,
    max_tokens=1000  # Increased for structured output
)

# Create agent with ToolStrategy for structured output
agent = create_agent(
    model=model,
    tools=[],  # No tools needed for extraction
    response_format=ToolStrategy(ContactInfo)
)

# Invoke the agent
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"
        }
    ]
})

# Access the structured response
print(result["structured_response"])
# Output: ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

### ProviderStrategy 提供者策略

`ProviderStrategy` 使用模型提供者的原生结构化输出生成。这更可靠，但仅适用于支持原生结构化输出的提供者（例如，OpenAI）：

```python wrap theme={null}
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-4o",
    response_format=ProviderStrategy(ContactInfo)
)
```

> 截至 `langchain 1.0` ，仅传递模式（例如 `response_format=ContactInfo` ）不再受支持。您必须明确使用 `ToolStrategy` 或 `ProviderStrategy` 。

**`ProviderStrategy` 是指 LLM 提供商（如 OpenAI、Claude 等）在其 API 中**原生支持**的结构化输出功能，而不是通过工具调用来实现。**要理解 `ProviderStrategy`，先理解它和 `ToolStrategy` 的区别：

#### ToolStrategy（工具调用策略）

```python
# LangChain 创建一个虚拟工具，让模型通过调用这个工具来返回结构化数据
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="deepseek-chat",  # 任何支持工具调用的模型
    response_format=ToolStrategy(ContactInfo)
)
```

模型的工作流程：

1. 模型分析输入
2. 模型决定调用一个名为 ContactInfo 的"工具"
3. 工具返回结构化数据
4. LangChain 提取结构化数据

**工作原理：**

- LangChain 在幕后创建一个虚拟工具
- 模型像调用真实工具一样调用它
- 返回的数据被验证和提取

**优点：** 几乎所有现代 LLM 都支持

#### ProviderStrategy（提供商原生策略）

```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-4o",  # 只有支持的提供商的模型
    response_format=ProviderStrategy(ContactInfo)
)
```

模型的工作流程：

1. 模型收到一个特殊的格式指令（JSON Schema）
2. 模型按照 OpenAI 的原生结构化输出 API 直接返回结构化数据
3. LangChain 提取数据

**工作原理：**

- 直接利用 OpenAI、Claude 等的原生 API 功能
- 模型不需要通过"工具调用"的中间层
- API 本身保证返回的数据符合 schema

```python
# 输入
text = "John Doe, john@example.com, (555) 123-4567"

# ========== ToolStrategy ==========
# LangChain 发送给模型的提示：
# "你需要调用一个叫 ContactInfo 的工具，参数是 {name, email, phone}"
#
# 模型的响应格式：
# {
#   "tool": "ContactInfo",
#   "arguments": {
#     "name": "John Doe",
#     "email": "john@example.com",
#     "phone": "(555) 123-4567"
#   }
# }

# ========== ProviderStrategy ==========
# OpenAI API 接收特殊参数：
# response_format: {
#   "type": "json_schema",
#   "json_schema": {
#     "type": "object",
#     "properties": {
#       "name": {"type": "string"},
#       "email": {"type": "string"},
#       "phone": {"type": "string"}
#     }
#   }
# }
#
# 模型的响应格式（直接返回）：
# {
#   "name": "John Doe",
#   "email": "john@example.com",
#   "phone": "(555) 123-4567"
# }
```

#### 各提供商支持情况

| 提供商           | ProviderStrategy 支持 | 优先级          |
| ---------------- | --------------------- | --------------- |
| OpenAI           | 支持（JSON Mode）     | 推荐使用        |
| Anthropic Claude | 支持                  | 推荐使用        |
| Gemini           | 支持                  | 推荐使用        |
| DeepSeek         | 不支持                | 用 ToolStrategy |
| Grok             | 支持                  | 推荐使用        |
| 其他大多数模型   | 不支持                | 用 ToolStrategy |

#### ProviderStrategy 的优势

##### 更可靠
ProviderStrategy 通过模型 API 直接保证,返回的数据 100% 符合 schema,几乎不会出现格式错误


##### 性能更好
ToolStrategy 需要额外的 token 消耗,ProviderStrategy 利用原生 API，token 效率更高

ToolStrategy 消耗：
- 发送工具定义的 token
- 模型调用工具的 token
- 返回结果的 token
总共可能多花 50-100 个 token

ProviderStrategy 消耗：只需要在系统提示中包含 schema,节省很多 token


##### 成本更低
由于 token 效率高，使用成本也更低官方文档提到"消除额外 LLM 调用的成本"

#### 实际使用建议

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

# ========== 如果用 OpenAI/Claude ==========
agent = create_agent(
    model="gpt-4o",  # 或 "claude-3-5-sonnet-20241022"
    response_format=ProviderStrategy(ContactInfo)  # 推荐
)

# ========== 如果用其他模型（如 DeepSeek）==========
agent = create_agent(
    model="deepseek-chat",
    response_format=ToolStrategy(ContactInfo)  # 必须用 ToolStrategy
)

# ========== 自动选择（推荐做法）==========
# LangChain 会自动检测：
# - 如果模型支持 ProviderStrategy → 自动使用
# - 如果模型不支持 → 自动降级到 ToolStrategy

agent = create_agent(
    model="gpt-4o",
    response_format=ContactInfo  # 直接传 schema，LangChain 自动选择最优方案
)
```


| 特点           | ProviderStrategy              | ToolStrategy                |
| -------------- | ----------------------------- | --------------------------- |
| **工作原理**   | 模型原生支持                  | 工具调用模拟                |
| **支持的模型** | OpenAI, Claude, Gemini 等少数 | 几乎所有模型                |
| **可靠性**     | 更高（API 保证）              | 中等（需要模型遵循）        |
| **性能**       | 更快，token 效率高            | 标准                        |
| **成本**       | 更低                          | 更高                        |
| **什么时候用** | 有支持的模型时推荐            | 模型不支持 ProviderStrategy |

### Memory 记忆

代理通过消息状态自动维护对话历史。您还可以配置代理使用自定义状态模式，在对话过程中记住额外信息。状态中存储的信息可以被视为代理的[短期记忆](https://docs.langchain.com/oss/python/langchain/short-term-memory)：

自定义状态模式必须扩展[`AgentState`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.AgentState)作为`TypedDict`。

定义自定义状态有两种方法：

1. 通过 [中间件](https://docs.langchain.com/oss/python/langchain/middleware)（推荐） 
2. 通过 [`state_schema`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware.state_schema) 在 [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent) 上

> 通过中间件定义自定义状态比通过 `state_schema` 在 `create_agent` 上定义更受推荐，因为它允许你将状态扩展在概念上局限于相关的中间件和工具。`state_schema` 在 `create_agent` 上仍然支持以保持向后兼容性。

#### Defining state via middleware 通过中间件定义状态
使用中间件来定义自定义状态，当您的自定义状态需要被特定中间件钩子和工具访问时。

```python
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
```
```
Test 1: Technical user with detailed preferences
✓ User Preferences Applied: style=technical, verbosity=detailed
  Call #1
✓ User Preferences Applied: style=technical, verbosity=detailed
  Call #2

✓ Final Response:
Now, about machine learning:

**Machine Learning** is a subset of artificial intelligence that focuses on developing algorithms and statistical models that enable computers to perform tasks without being explicitly programmed for every scenario. Instead, these systems learn from data and improve their performance over time.

Key aspects of machine learning include:

- **Learning from data**: ML algorithms analyze patterns in data to make predictions or decisions
- **Adaptive improvement**: The models get better as they're exposed to more data
- **Automated pattern recognition**: They can identify complex patterns that might be difficult for humans to detect

Common types of machine learning:
- **Supervised learning**: Training with labeled data (like classification and regression)
- **Unsupervised learning**: Finding patterns in unlabeled data (like clustering)
- **Reinforcement learning**: Learning through trial and error with rewards/punishments

Machine learning is used in many applications today, including recommendation systems, image recognition, natural language processing, fraud detection, and autonomous vehicles.

And regarding the weather in Beijing - it's currently sunny and 22°C, which sounds like pleasant weather!

Test 2: Simple user with brief preferences
✓ User Preferences Applied: style=simple, verbosity=brief
  Call #1
✓ User Preferences Applied: style=simple, verbosity=brief
  Call #2

✓ Final Response:
2 + 2 = 4

Test 3: Default user (no preferences)
✓ User Preferences Applied: style=default, verbosity=normal
  Call #1
✓ User Preferences Applied: style=default, verbosity=normal
  Call #2

✓ Final Response:
10 * 5 equals 50.
```

#### Defining state via `state_schema` 通过 `state_schema` 定义状态
使用 [`state_schema`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware.state_schema) 参数作为快捷方式来定义仅在工具中使用的自定义状态。

```python
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
```
```
✓ Success!
Response: I understand you prefer technical explanations. However, I'm currently limited in my ability to provide detailed technical information as the available tools are quite basic. The only function I have access to right now is a simple information retrieval tool that doesn't require any parameters.

If you have specific technical questions or topics you'd like me to explain, I'd be happy to provide technical explanations based on my general knowledge. Just let me know what particular subject or concept you're interested in, and I'll do my best to give you a thorough technical breakdown.

What technical area would you like to explore?
User Preferences: {'style': 'technical', 'verbosity': 'detailed'}
```

> 截至 `langchain 1.0` ，自定义状态模式必须为 `TypedDict` 类型。Pydantic 模型和数据类不再受支持。有关更多详细信息，请参阅 [v1 迁移指南]([LangChain v1 migration guide - Docs by LangChain](https://docs.langchain.com/oss/python/migrate/langchain-v1#state-type-restrictions))。

### Streaming 流式传输

我们已经看到如何使用 `invoke` 调用代理以获得最终响应。如果代理执行多个步骤，这可能需要一些时间。为了显示中间进度，我们可以随着消息的发生而流式传输消息。

```python
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]
}, stream_mode="values"):
    # Each chunk contains the full state at that point
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
```
> 关于流式传输的更多详情，请参阅[Streaming](https://docs.langchain.com/oss/python/langchain/streaming)。

### Middleware 中间件

[中间件](https://docs.langchain.com/oss/python/langchain/middleware) 为自定义执行不同阶段的代理行为提供了强大的扩展性。您可以使用中间件来：

* 调用模型前的处理状态（例如，消息修剪、上下文注入）

* 修改或验证模型的响应（例如，安全护栏、内容过滤）

* 使用自定义逻辑处理工具执行错误

* 根据状态或上下文实现动态模型选择

* 添加自定义日志记录、监控或分析

中间件无缝集成到代理的执行图中，允许您在关键点拦截和修改数据流，而无需更改核心代理逻辑。

> 有关包括 [`@before_model`]([Middleware | LangChain Reference](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.before_model)) 、 [`@after_model`]([Middleware | LangChain Reference](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.after_model)) 和 [`@wrap_tool_call`]([Middleware | LangChain Reference](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_tool_call)) 等装饰器的全面中间件文档，请参阅 [Middleware]([Overview - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/middleware/overview))。

