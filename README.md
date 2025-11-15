<div align="center">
<!-- Title: -->
<img src="https://s2.loli.net/2025/11/15/KHgQ6NPjEUAuVZh.png">
<h1><div align="center">ChainReaction</h1>
<a href="https://github.com/shanchuann/ChainReaction"><img src="https://img.shields.io/badge/LangChain-Learning-blue?style=flat-square" height="20" alt="LangChain Learning"></a>
<a href="https://github.com/shanchuann/ChainReaction/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" height="20" alt="MIT License"></a>
<a href="https://github.com/shanchuann/ChainReaction/blob/main/requirements.txt"><img src="https://img.shields.io/badge/Python-3.7%2B-yellow?style=flat-square" height="20" alt="Python 3.7+"></a>
</div>

  <h2><div align="center">专注于 LangChain 框架的学习与实践仓库，提供中文文档和可运行示例</h2>


### 仓库结构

```
ChainReaction/
├── Install/           # LangChain 安装指南
├── Quickstart/        # 快速入门示例（基础到进阶智能体实现）
├── LangChain exercise/ # 特定场景练习
├── requirements.txt   # 项目依赖清单
├── LICENSE            # MIT 许可证
└── .gitignore         # Git 版本控制忽略文件配置
```

### 核心内容

#### 安装指南
提供 LangChain 在本地环境的详细安装步骤，包括：

系统要求：Python 3.7+、pip 及支持的操作系统（Windows/macOS/Linux）

安装步骤：
1. 安装 Python（推荐最新版本）
2. 创建并激活虚拟环境（可选但推荐）
```bash
python -m venv langchain-env
# Linux/macOS 激活
source langchain-env/bin/activate
# Windows 激活
langchain-env\Scripts\activate
```
3. 安装 LangChain 核心库及常用集成
```bash
pip install langchain
# 示例：安装 OpenAI/Anthropic 集成
pip install -U langchain-openai langchain-anthropic
```
4. 验证安装：通过命令查看版本号确认安装成功
```bash
python -c "import langchain; print(langchain.__version__)"
```

- 后续学习建议：阅读官方文档、探索示例项目、加入社区交流

#### 快速入门
通过实际代码示例，从基础到进阶构建 AI 智能体，包括：
- **基础智能体**：使用 DeepSeek 模型，定义简单天气查询工具，通过 `create_agent` 实现基本交互
  ```python
  # 关键代码片段
  from langchain_openai import ChatOpenAI
  from langchain.agents import create_agent
  
  model = ChatOpenAI(
      api_key=os.getenv('DEEPSEEK_API_KEY'),
      base_url="https://api.deepseek.com/v1",
      model="deepseek-chat"
  )
  
  agent = create_agent(model=model, tools=[get_weather], system_prompt="You are a helpful assistant")
  ```
- **进阶天气预报代理**：
  - 支持双关语响应的专业角色设定
  - 多工具调用（获取位置、查询天气）
  - 对话记忆功能（通过 `InMemorySaver` 保持上下文）
  - 结构化响应格式（定义 `ResponseFormat` 确保输出一致性）
  - 上下文管理（通过 `Context` 传递用户 ID 等信息）

#### 练习示例
提供特定场景的实践案例，例如：
- **情人节玫瑰宣传语生成**：使用 LangChain 集成 Ollama 本地模型（如 `qwen3-vl:8b`），生成创意宣传语
  ```python
  from langchain_ollama import ChatOllama
  
  llm = ChatOllama(
      model="qwen3-vl:8b",
      base_url="http://localhost:11434"
  )
  text = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
  print(text)
  ```

### 依赖说明
核心依赖已在 `requirements.txt` 中定义，主要包括：
```
langchain~=1.0.7          # LangChain 核心框架
langchain-openai~=1.0.3   # OpenAI/DeepSeek 等模型集成
python-dotenv~=1.2.1      # 环境变量管理
```

### 许可证
本项目采用 MIT 许可证，详情参见 [LICENSE](LICENSE)。
