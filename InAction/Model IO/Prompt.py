# 导入LangChain框架中的提示模板模块
from langchain_core.prompts import PromptTemplate # 定义原始模板字符串，明确提示逻辑与变量占位
template = """您是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""
# 通过PromptTemplate类的from_template方法，将原始字符串模板转化为LangChain提示模板对象
prompt = PromptTemplate.from_template(template)
# 打印提示模板对象，查看其核心属性配置
print(prompt)