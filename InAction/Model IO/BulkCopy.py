# 1. 复用提示模板（与前文一致，无需重复定义）
# 导入LangChain框架中的提示模板模块
from langchain_core.prompts import PromptTemplate # 定义原始模板字符串，明确提示逻辑与变量占位

template = """您是一位专业的鲜花店文案撰写员。\n对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？"""
prompt = PromptTemplate.from_template(template)

# 2. 配置API密钥与实例化模型（复用单条生成逻辑）
import os
from langchain_openai import ChatOpenAI

# 3. 实例化大语言模型（指定模型版本）
model = ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat"
)

# 3. 准备批量数据（鲜花名称与对应售价列表）
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 4. 循环生成批量文案
for flower, price in zip(flowers, prices):
    # 模板实例化（动态注入当前鲜花数据）
    input_prompt = prompt.format(flower_name=flower, price=price)
    # 模型调用
    output = model.invoke(input_prompt)
    # 输出结果
    print(output)