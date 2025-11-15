# 安装 LangChain

本指南将帮助您在本地环境中安装和配置 LangChain，以便您可以开始构建和运行基于 LangChain 的应用程序。

## 系统要求

- Python 3.7 或更高版本
- pip（Python 包管理器）
- 一个支持的操作系统（Windows、macOS 或 Linux）

## 安装步骤
1. **安装 Python**  
   如果您尚未安装 Python，请访问 [Python 官方网站](https://www.python.org/downloads/) 下载并安装最新版本的 Python。
2. **创建虚拟环境（可选）**  
   建议为您的项目创建一个虚拟环境，以隔离依赖项。您可以使用以下命令创建和激活虚拟环境：
   ```bash
   python -m venv langchain-env
   source langchain-env/bin/activate  # 在 Windows 上使用 `langchain-env\Scripts\activate`
   ```
3. **安装 LangChain**
   使用 pip 安装 LangChain：
   ```bash
   pip install langchain
   ```
4. **验证安装**
   安装完成后，您可以通过以下命令验证 LangChain 是否安装成功：
   ```bash
   python -c "import langchain; print(langchain.__version__)"
   ```
   如果没有错误并且显示了版本号，则表示安装成功。

LangChain 提供与数百个 LLM 和数千个其他集成的集成。这些存在于独立的提供商包中。例如：
```bash
# Installing the OpenAI integration
pip install -U langchain-openai

# Installing the Anthropic integration
pip install -U langchain-anthropic
```

## 接下来做什么？
- 阅读 [LangChain 文档](https://langchain.com/docs/) 以了解如何使用 LangChain 构建应用程序。
- 探索示例项目和教程，以加深对 LangChain 的理解。
- 加入 LangChain 社区，获取支持并与其他开发者交流。
- 开始构建您的第一个 LangChain 应用程序！
