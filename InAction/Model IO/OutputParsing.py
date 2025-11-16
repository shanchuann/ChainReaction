# 配置OpenAI API密钥（注入环境变量）
import os
# os.environ["OPENAI_API_KEY"] = "你的OpenAI API Key"

# 导入LangChain核心组件
from langchain_core.prompts import PromptTemplate # 定义原始模板字符串，明确提示逻辑与变量占位
# from langchain_openai import OpenAI  # OpenAI模型接口
from langchain_ollama import ChatOllama # Import ChatOllama
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
import pandas as pd  # 数据处理与存储

# 定义响应模式（规范输出字段的名称与含义）
response_schemas = [
    ResponseSchema(
        name="description",
        description="鲜花的描述文案，需简洁且具有吸引力"
    ),
    ResponseSchema(
        name="reason",
        description="解释该文案的创作逻辑或设计思路"
    )
]

# 基于响应模式创建结构化输出解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 定义原始提示模板（包含格式说明占位符）
prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，请提供：
1. 吸引人的简短描述文案；
2. 该文案的创作理由。
{format_instructions}"""  # 格式说明占位符

# 获取解析器生成的格式约束说明（指导模型输出结构）
format_instructions = output_parser.get_format_instructions()

# 创建最终提示模板（整合原始模板与格式说明）
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["flower_name", "price"],  # 动态变量：鲜花名称、价格
    partial_variables={"format_instructions": format_instructions}  # 固定变量：格式说明
)

# 实例化模型
model=ChatOllama(
        model="llama2:latest",
        base_url="http://localhost:11434"  # 默认地址
    )

# 准备批量处理数据（鲜花名称与对应价格）
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 初始化DataFrame用于存储结果（定义字段：鲜花名称、价格、文案、理由）
df = pd.DataFrame(columns=["flower", "price", "description", "reason"])

# 循环生成并解析文案
for flower, price in zip(flowers, prices) :
    # 基于模板生成模型输入（注入当前鲜花名称与价格）
    input_prompt = prompt.format(flower_name=flower, price=price)

    # 调用模型获取原始输出（非结构化文本）
    output = model.invoke(input_prompt)
    # 解析原始输出为结构化字典（匹配response_schemas定义）
    parsed_output = output_parser.parse(output.content)
    # 补充鲜花名称与价格字段
    parsed_output["flower"] = flower
    parsed_output["price"] = price

    # 将结构化数据添加到DataFrame
    df.loc[len(df)] = parsed_output

# 打印结构化数据（字典格式）
print("结构化输出结果：\n", df.to_dict(orient="records"))

# 保存为CSV文件（便于后续分析或系统集成）
df.to_csv("flowers_with_descriptions.csv", index=False)