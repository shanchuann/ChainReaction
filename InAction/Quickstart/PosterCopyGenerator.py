import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

# ---- Part I 初始化图像字幕生成模型
# 指定使用的HuggingFace图像字幕生成模型
hf_model = "Salesforce/blip-image-captioning-large"
# 初始化处理器，用于预处理图像数据
processor = BlipProcessor.from_pretrained(hf_model, use_fast=True)
# 初始化BLIP模型，用于生成图像描述
blip_model = BlipForConditionalGeneration.from_pretrained(hf_model)


# ---- Part II 定义图像字幕生成工具类
class ImageCapTool(BaseTool) :
    name: str = "image_captioner"  # 工具名称，用于代理识别
    description: str = "为图片创作说明文案"  # 工具描述，告诉代理此工具的功能

    def _run(self, url: str) -> str :
        """
        工具的核心方法：根据图片URL生成文字描述

        参数:
            url: 图片的URL地址

        返回:
            caption: 生成的图片描述文字
        """
        # 从URL下载图片并转换为RGB格式
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        # 使用处理器预处理图片，准备模型输入
        inputs = processor(image, return_tensors="pt")
        # 使用BLIP模型生成图片描述，限制最大生成长度
        out = blip_model.generate(**inputs, max_new_tokens=20)
        # 解码模型输出，获取可读的文字描述
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption


# ---- Part III 初始化并运行LangChain智能代理
# 初始化Ollama语言模型，连接到本地服务
llm = ChatOllama(model="qwen3-vl:8b", base_url="http://localhost:11434")

# 创建智能代理，结合语言模型和工具
agent = create_agent(
    model=llm,  # 使用的语言模型
    tools=[ImageCapTool()],  # 可用的工具列表
    system_prompt="You are a helpful assistant"  # 系统提示，定义代理的角色
)

# 指定要分析的图片URL
img_url = 'https://smms.app/image/Ovn8ufwPmWh9VpF'
# 调用代理处理用户请求：为图片生成中文推广文案
result = agent.invoke({"messages" : [{"role" : "user", "content" : f"{img_url}\n请创作合适的中文推广文案"}]})

# 打印结果
print(result)