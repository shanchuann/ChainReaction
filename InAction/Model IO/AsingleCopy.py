# # 1. 配置OpenAI API密钥（环境变量注入方式）
import os
# os.environ["OPENAI_API_KEY"] = "你的OpenAI API Key"

# 2. 导入LangChain的Ollama模型接口模块
# from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI

# 3. 实例化大语言模型（指定模型版本）
model = model = ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat"
)

# 4. 基于已定义的提示模板生成具体输入（模板复用前文“鲜花文案”模板）
input_prompt = prompt.format(flower_name="玫瑰", price="50")

# 5. 调用模型接口获取输出
output = model.invoke(input_prompt)

# 6. 输出结果
print(output)