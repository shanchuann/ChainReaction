## 情人节玫瑰宣传语

情人节到啦，你的花店需要推销红色玫瑰，怎样通过LangChain来用程序的方式给咱们生成简短的宣传语呢？

很简单，你需要:
安装两个包

- 通过 `pip install langchain` 来安装 LangChain

- 通过 `pip install -qU langchain-ollama` 以便在 LangChain 中使用 Ollama 的模型。

另外，需要在 Ollama 中下载一个合适的模型，比如 qwen3-vl:8b。你可以通过 `ollama list` 来查看已经下载的模型，通过 `ollama pull qwen3-vl:8b` 来下载这个模型。

接下来就是编写代码了。下面的代码展示了如何使用 LangChain 和 Ollama 来生成情人节玫瑰的宣传语：

```python
import os
from langchain_ollama import ChatOllama # Import ChatOllama

llm = model=ChatOllama(
        model="qwen3-vl:8b",
        base_url="http://localhost:11434"  # 默认地址
    )
text = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
print(text)
```

运行这段代码后，你会得到一条简短而有吸引力的情人节红玫瑰宣传语。这样，你就成功地利用 LangChain 和 Ollama 来实现了一个简单的应用场景。

## 海报文案生成器

你已经制作好了一批鲜花的推广海报，想为每一个海报的内容，写一两句话

我们可以LangChain代理工具来实现这个功能

这段代码主要包含三个部分：

1. 初始化图像字幕生成模型（HuggingFace中的image-caption模型）。
2. 定义LangChain图像字幕生成工具。
3. 初始化并运行LangChain Agent（代理），这个Agent是OpenAI的大语言模型，会自动进行分析，调用工具，完成任务。

不过，这段代码需要的包比较多。在运行这段代码之前，你需要先更新LangChain到最新版本，安装HuggingFace的Transformers库（开源大模型工具），并安装 Pillow（Python图像处理工具包）和 PyTorch（深度学习框架）。
```bash
pip install --upgrade langchain
pip install transformers
pip install pillow
pip install torch torchvision torchaudio
```
```python
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
class ImageCapTool(BaseTool):
    name: str = "image_captioner"  # 工具名称，用于代理识别
    description: str = "为图片创作说明文案"  # 工具描述，告诉代理此工具的功能

    def _run(self, url: str) -> str:
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
result = agent.invoke({"messages": [{"role": "user", "content": f"{img_url}\n请创作合适的中文推广文案"}]})

# 打印结果
print(result)
```

根据输入的图片URL，由大语言模型驱动的LangChain Agent，首先利用图像字幕生成工具将图片转化为字幕，然后对字幕做进一步处理，生成中文推广文案。

