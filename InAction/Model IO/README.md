模型作为LangChain框架的底层核心组件，是构建基于语言模型（LLM）应用的关键要素。从技术本质来看，LangChain应用开发是以LangChain为技术框架，通过API调用大语言模型（LLM）解决特定业务问题的实现过程。

可以说，LangChain框架的整体运行逻辑均以LLM为核心驱动力。若脱离模型支撑，LangChain框架将丧失其核心应用价值，其存在的意义也无从谈起。


## Model I/O

对模型的使用流程可拆解为三个核心环节，分别是**输入提示（Format）**、**模型调用（Predict）** 与**输出解析（Parse）**。上述三个环节构成有机整体，在LangChain框架中，该过程被统一定义为**Model I/O（Input/Output，即输入/输出）**。

![image-20251116210701867](https://s2.loli.net/2025/11/16/Gfd6IZWKPM5kF1o.png)

在Model I/O的各环节中，LangChain均提供标准化模板与工具组件，可快速构建适配各类语言模型的调用接口，降低模型集成与应用开发的技术门槛。具体环节功能如下：

1. **提示模板**：作为模型使用的首个环节，需向模型输入提示信息。基于LangChain可构建标准化提示模板，该模板能够根据实际业务需求动态选择输入内容，并针对特定任务与应用场景对输入信息进行精准适配调整，确保提示信息与模型能力、业务目标高度匹配。

2. **语言模型**：LangChain支持通过通用接口实现对语言模型的调用。这一特性意味着，无论开发过程中选用何种类型的语言模型，均能通过统一的调用方式完成交互，无需针对不同模型单独开发适配逻辑，显著提升了开发过程的灵活性与便捷性。

3. **输出解析**：LangChain同时提供模型输出信息提取功能。借助输出解析器，开发人员可从模型输出结果中精准提取核心需求信息，无需处理冗余或无关数据；更重要的是，其能够将大语言模型返回的非结构化文本，转换为程序可直接处理的结构化数据（如JSON、表格等格式），为后续业务逻辑的自动化处理提供关键支撑。

## 提示模板

在大语言模型（LLM）的应用实践中，高质量的提示是确保模型输出贴合需求、提升响应精准度的关键前提。结合行业实践与技术沉淀，构建有效提示的核心逻辑可提炼为两大核心原则，二者共同作用于 “让模型精准理解需求、高效输出结果” 的目标：

**原则一：传递清晰、明确的指令**

模型对需求的理解依赖于提示中指令的明确性 —— 模糊的表述易导致模型误判任务边界，或输出偏离预期的内容。因此，构建提示时需聚焦 “降低模型理解成本”，具体可通过以下方式落地：

- **明确任务定位**：清晰告知模型需扮演的角色（如 “专业产品文案师”“技术文档译者”）、核心任务（如 “撰写产品卖点”“解释技术概念”），避免模型因角色或任务模糊而输出泛化内容；
- **界定输出标准**：若需特定格式（如列表、表格、JSON）、篇幅（如 “300 字以内”“分 3 点阐述”）或风格（如 “正式商务风”“口语化科普风”），需在提示中明确标注，减少后续对输出的调整成本；
- **补充关键约束**：针对任务中的核心限制条件（如 “避免使用专业术语”“需包含 XX 数据”）进行明确说明，防止模型忽略关键需求而产生无效输出。

**原则二：引导模型进行逐步、严谨的推理**

面对复杂任务（如逻辑分析、问题拆解、多步骤计算），直接要求模型输出最终结果易导致推理跳跃、漏洞或错误。此时需通过提示引导模型 “慢思考”，模拟人类解决复杂问题的分步逻辑，具体路径包括：

- **拆解任务步骤**：将复杂任务拆分为可执行的子步骤，让模型按步骤推导（如 “先分析用户需求痛点，再对应产品功能提出解决方案，最后总结核心价值”）；
- **要求展示推理过程**：在提示中明确 “需先说明推理思路，再给出最终结论”，例如解决数学问题时，引导模型先列出公式与计算步骤，再输出结果，既便于验证逻辑正确性，也能减少模型 “凭直觉输出” 的误差；
- **提供中间锚点**：对于极复杂任务，可在提示中给出部分中间结论或参考方向（如 “在分析市场趋势时，需重点考虑政策影响与用户需求变化两个维度”），帮助模型锚定推理方向，避免偏离核心逻辑。

这两大原则并非孤立存在：“清晰指令” 为模型划定了 “做什么、怎么做” 的边界，“引导逐步推理” 则为模型提供了 “如何做好” 的路径。二者结合，可大幅降低模型输出的随机性，让响应更贴合实际应用场景的需求。

### 提示模板构建示例

以 “为鲜花销售场景生成产品简介文案” 为例，需构建适配的提示模板：当员工或顾客查询特定鲜花信息时，调用该模板可快速生成符合营销需求的描述文本。其技术实现流程如下：

```plain
# 导入LangChain框架中的提示模板模块
from langchain_core.prompts import PromptTemplate # 定义原始模板字符串，明确提示逻辑与变量占位
template = """您是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""
# 通过PromptTemplate类的from_template方法，将原始字符串模板转化为LangChain提示模板对象
prompt = PromptTemplate.from_template(template)
# 打印提示模板对象，查看其核心属性配置
print(prompt)
```

#### 模板核心属性解析

上述代码生成的 PromptTemplate 对象，其具体内容如下：

```plain
input_variables=['flower_name', 'price'] input_types={} partial_variables={} template='您是一位专业的鲜花店文案撰写员。\n\n对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？\n'
```

- **模板格式与变量**：该模板以 “f-string” 为格式标准，包含`{flower_name}`（鲜花名称）与`{price}`（售价）两个输入变量（input_variables），二者作为占位符，在实际调用时将被具体业务数据（如 “玫瑰”“99”）替换，实现提示的动态适配；
- **核心方法作用**：`from_template`是 PromptTemplate 类的核心类方法，支持直接从字符串模板生成功能完备的 PromptTemplate 对象，无需手动配置所有属性，简化开发流程；
- **辅助配置项**：`output_parser`（输出解析器）本例中未指定，故为 None；`validate_template=True`表示启用模板验证机制，确保模板语法合规、变量配置无误；`partial_variables`（部分变量）暂为空，支持后续按需预设部分固定值。

LangChain 框架围绕提示模板提供了多类专用类、函数及工具，并针对客服对话、内容生成、数据分析等**典型应用场景设计了丰富的内置模板**。这些资源可大幅降低提示构建的技术门槛，帮助开发者快速实现 “模板复用” 与 “场景适配”，提升大语言模型应用的开发效率。

## 语言模型体系与调用实践

在LangChain框架中，语言模型作为Model I/O环节的核心执行单元，其体系覆盖了不同功能定位的模型类型，可满足多样化的业务场景需求。同时，LangChain通过标准化的调用逻辑与工具封装，显著提升了模型应用的效率与灵活性。


### LangChain支持的三类核心语言模型

LangChain针对不同任务场景，对语言模型进行了清晰分类，三类核心模型的功能定位、输入输出格式及典型示例如下：

| 模型类型                            | 功能定位                                                     | 输入/输出格式                                                | 典型示例                                                   |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------- |
| **大语言模型（LLM/Text Model）**    | 处理通用文本生成、理解任务，适用于无固定交互格式的文本需求   | 输入：文本字符串<br>输出：文本字符串                         | OpenAI text-davinci-003、Meta LLaMA、Anthropic Claude      |
| **聊天模型（Chat Model）**          | 适配对话类场景，支持结构化交互，模拟多轮对话逻辑             | 输入：聊天消息列表（含角色、内容等结构化信息）<br>输出：聊天消息（含角色与响应内容） | OpenAI ChatGPT系列（gpt-3.5-turbo、gpt-4）                 |
| **文本嵌入模型（Embedding Model）** | 将文本转化为高维向量（Embedding），用于文本相似度计算、向量数据库存储，支撑检索增强等场景 | 输入：文本字符串<br>输出：浮点数向量列表                     | OpenAI text-embedding-ada-002、Hugging Face BERT Embedding |

需说明的是，文本嵌入模型核心作用于“文本向量化”环节，与本次聚焦的“提示工程与文案生成”场景关联度较低，后续内容将以大语言模型的调用实践为核心展开。

以“生成鲜花产品文案”为典型场景，通过LangChain调用大语言模型的完整流程如下，涵盖API配置、模型实例化、提示模板适配及结果输出全环节：

#### 单条文案生成实现

```plain
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
```

```
content='【50元心动玫瑰】 \n不凋零的浪漫，只为值得的时刻绽放 \n精选A级红玫瑰，丝绒花瓣包裹炽热爱意 \n每一支皆由花艺师亲手修剪 \n让50元的价值 \n在拆开包装的瞬间 \n化作她眼底的星光 \n· 周一清晨的办公桌惊喜 \n· 晚餐时突然变出袖口的温柔 \n· 纪念日里不说抱歉的拥抱 \n今日订购即赠手写卡片 \n让价格成为秘密，让爱意成为宣告' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 108, 'prompt_tokens': 32, 'total_tokens': 140, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 32}, 'model_provider': 'openai', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_ffc7281d48_prod0820_fp8_kvcache', 'id': '05b91e36-c303-4adc-885d-83a8ef72452f', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--263f3b8b-34b7-4019-9c1e-c4e0159bf775-0' usage_metadata={'input_tokens': 32, 'output_tokens': 108, 'total_tokens': 140, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}
```

- **提示模板实例化**：通过`prompt.format()`方法，将模板中的占位符`{flower_name}`与`{price}`替换为具体值（“玫瑰”“50”），生成符合模型输入要求的文本指令，最终指令为：“您是一位专业的鲜花店文案撰写员。对于售价为 50 元的玫瑰，您能提供一个吸引人的简短描述吗？”
- **模型调用**：通过`model.invoke()`方法发起同步调用，LangChain已封装底层API交互逻辑，无需手动处理请求参数拼接、响应解析等细节。
- **典型输出结果**：`让你心动！50元就可以拥有这支充满浪漫气息的玫瑰花束，让TA感受你的真心爱意。`


#### 批量文案生成实现

基于同一提示模板，可快速扩展至多类鲜花的批量文案生成，核心逻辑为通过循环遍历数据列表，复用模板与模型实例：

```plain
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
```

```
content='【50元心动玫瑰】 \n✨ 不凋谢的浪漫 只为悦己盛放 ✨\n\n精选A级红玫瑰·每日限定\n每一瓣都裹着晨曦的露水感\n丝绒质感花瓣与哑光绿枝的碰撞\n—— 比口红更提气色，比情话更触手可及\n\n▫️办公室治愈系能量站\n▫️闺蜜暗语小花束\n▫️自购生活仪式感\n（搭配雾面英文报包装 解锁电影感日常）\n\n#50元把高级感带回家 #玫瑰无需等节日' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 126, 'prompt_tokens': 32, 'total_tokens': 158, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 32}, 'model_provider': 'openai', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_ffc7281d48_prod0820_fp8_kvcache', 'id': '66598d8b-debc-486f-82dc-e6816d80e40e', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--93efa155-58ea-43d2-a57c-1d73034cbffb-0' usage_metadata={'input_tokens': 32, 'output_tokens': 126, 'total_tokens': 158, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}
content='【清雅百合 · 30元带走一室芬芳】  \n✨ 晨露浸润的纯白花瓣，悄然绽放温柔弧度  \n✨ 每支含2-3个花头 花期长达10天  \n✨ 搭配翠绿茎叶 治愈系天然香氛  \n🎯 办公室点睛/书房雅趣/赠人佳品  \n「30元解锁春日柔软时光」' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 83, 'prompt_tokens': 32, 'total_tokens': 115, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 32}, 'model_provider': 'openai', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_ffc7281d48_prod0820_fp8_kvcache', 'id': '791f73bd-b18e-40c9-8843-fd21813e40e7', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--336a64b7-4b17-4d62-8bb6-fdd76ea9d311-0' usage_metadata={'input_tokens': 32, 'output_tokens': 83, 'total_tokens': 115, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}
content='【暖爱如馨 · 康乃馨】  \n20元，把温柔捧在掌心！这束康乃馨瓣瓣饱满，缀着晨露般的生机，粉糯色泽如母亲微笑的温度。不论赠予挚爱、致敬恩师，或为自己点亮一隅春光，它都在轻声说：你值得被美好簇拥。✨' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 75, 'prompt_tokens': 34, 'total_tokens': 109, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 34}, 'model_provider': 'openai', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_ffc7281d48_prod0820_fp8_kvcache', 'id': '4e1b1fa9-f0c9-4324-939c-6c8a22f9bc94', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--665c1ef9-37f9-4e08-9e30-6c3edb40346a-0' usage_metadata={'input_tokens': 34, 'output_tokens': 75, 'total_tokens': 109, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}
```

- 玫瑰：`这支玫瑰，深邃的红色，传递着浓浓的深情与浪漫，令人回味无穷！`
- 百合：`百合：美丽的花朵，多彩的爱恋！30元让你拥有它！`
- 康乃馨：`康乃馨—20元，象征爱的祝福，送给你最真挚的祝福。`


### LangChain与直接调用API的核心差异的优势

在上述场景中，直接调用OpenAI API（非LangChain框架）也可实现类似功能，其核心代码如下：

```plain
import openai
# 配置API密钥
openai.api_key = "Your-OpenAI-API-Key"

# 定义字符串模板
prompt_text = "您是一位专业的鲜花店文案撰写员。对于售价为{}元的{}，您能提供一个吸引人的简短描述吗？"

# 批量生成逻辑
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]
for flower, price in zip(flowers, prices):
    # 字符串格式化生成提示
    prompt = prompt_text.format(price, flower)
    # 调用OpenAI Completions API
    response = openai.completions.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=100
    )
    # 解析输出结果
    print(response.choices[0].text.strip())
```

尽管直接调用API可实现基础功能，但LangChain的核心优势体现在**工程化效率提升**与**扩展性**上，具体可归纳为三点：

#### 模板的“一次定义，多场景复用”

LangChain的`PromptTemplate`对象将模板逻辑与数据注入解耦，只需定义一次模板结构，即可在不同模块、不同数据集中复用，无需重复编写字符串格式化逻辑。相比直接使用`f-string`或`{}`占位符的原生字符串，LangChain模板更便于维护（如统一修改模板风格、补充约束条件），尤其适用于大型项目中多模块共用提示逻辑的场景。

#### 内置功能的集成化支持

LangChain的`PromptTemplate`内置了三类关键功能，无需手动开发：

- **模板验证（validate_template）**：自动校验模板语法正确性（如占位符是否完整、格式是否合规），避免因字符串格式化错误导致的模型调用失败；
- **输出解析器（output_parser）**：可直接关联输出解析器，将模型返回的非结构化文本转化为JSON、列表等结构化数据，省去手动解析响应的步骤；
- **部分变量预设（partial_variables）**：支持预设部分固定变量（如“文案风格=正式营销风”），仅动态注入可变变量（如鲜花名称、价格），进一步简化调用逻辑。

#### 跨模型适配的“无感知切换”

若需将当前调用的“gpt-3.5-turbo-instruct”切换为其他模型（如Anthropic Claude、Meta LLaMA），LangChain无需修改提示模板与调用逻辑——仅需替换模型实例化代码（如`from langchain_anthropic import Anthropic`），即可实现模型切换。而直接调用API时，需针对不同模型的请求格式（如Anthropic的`messages`参数、OpenAI的`prompt`参数）重新适配代码，扩展性显著低于LangChain。

综上，LangChain并非简单封装模型API，而是通过“模板标准化”“功能集成化”“模型适配灵活化”，为语言模型的工程化应用提供了更高效、更可维护的解决方案，尤其适用于复杂场景下的模型应用开发。

从本次实践及前文案例可进一步提炼LangChain的核心价值：其定位类似机器学习领域的PyTorch、TensorFlow框架——**模型可按需选择（商业/开源、不同厂商）、自主优化（训练/微调），而调用模型的工程化逻辑（模板定义、参数传递、结果获取）具备标准化、可复用的章法**。

具体而言，LangChain结合提示模板的优势可归纳为五大工程化价值：

1. **提升代码可读性**：针对复杂提示或多变量场景，模板将提示文本与变量逻辑分离，结构更清晰，便于团队协作与代码复盘。
2. **强化逻辑可复用性**：模板可在多模块、多任务中直接调用，无需重复构造提示字符串，显著简化代码冗余。
3. **降低维护成本**：若需调整提示逻辑（如优化文案风格、补充约束条件），仅需修改模板本身，无需在代码中逐一查找所有调用节点，减少维护风险。
4. **简化变量处理**：自动完成多变量的注入与格式校验，避免手动字符串拼接导致的语法错误（如变量遗漏、格式错乱）。
5. **支持参数化生成**：基于同一模板，通过动态传入不同参数（如鲜花名称、价格、风格要求），可快速实现个性化文本生成，适配多样化业务需求。

## 输出解析

在基于语言模型的应用开发中，模型输出的结构化处理是衔接“自然语言响应”与“程序自动化处理”的关键环节。LangChain提供的输出解析器（Output Parser）功能，可将模型返回的非结构化文本高效转换为程序可直接处理的结构化数据（如字典、表格等），显著降低开发复杂度，提升应用落地效率。


### 从非结构化到结构化

在实际开发场景中，仅获取模型生成的文本字符串（如前文的鲜花文案）往往无法满足需求。多数应用需基于模型输出进行进一步计算、存储或展示，这要求输出数据具备明确的字段定义与格式规范。

以鲜花文案场景为例，若需同时获取“文案内容”与“创作理由”，模型可能返回如下非结构化文本（示例A）：  
**示例A**：“文案是：让你心动！50元就可以拥有这支充满浪漫气息的玫瑰花束，让TA感受你的真心爱意。为什么这样写呢？因为爱情是无价的，50元对应热恋中的情侣也会觉得值得。”  

此类输出需人工提取关键信息，无法直接被程序处理。而通过输出解析器，可将其转换为如下结构化数据（示例B）：  
**示例B**：  

```python
{
    "description": "让你心动！50元就可以拥有这支充满浪漫气息的玫瑰花束，让TA感受你的真心爱意。",
    "reason": "因为爱情是无价的，50元对应热恋中的情侣也会觉得值得。"
}
```

这种结构化转换实现了“模型输出”到“程序可用数据”的无缝衔接，为后续数据存储（如CSV、数据库）、逻辑处理（如条件判断）、可视化展示等环节提供了基础支撑。


### 基于LangChain的全流程实践

以下以“生成鲜花文案及创作理由，并存储为CSV文件”为例，详细说明LangChain输出解析器的使用流程，涵盖响应模式定义、解析器配置、提示模板整合及结果处理全环节。


#### 环境配置与依赖导入

首先完成OpenAI API密钥配置，并导入核心模块（提示模板、模型接口、输出解析器、数据处理工具等）：

```plain
# 配置OpenAI API密钥（注入环境变量）
import os
# os.environ["OPENAI_API_KEY"] = "你的OpenAI API Key"

# 导入LangChain核心组件
from langchain_core.prompts import PromptTemplate # 定义原始模板字符串，明确提示逻辑与变量占位
# from langchain_openai import OpenAI  # OpenAI模型接口
from langchain_ollama import ChatOllama # Import ChatOllama
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
import pandas as pd  # 数据处理与存储
```


#### 定义响应模式与输出解析器

通过`ResponseSchema`定义预期输出的字段结构（名称与描述），再基于该结构创建`StructuredOutputParser`，明确解析规则：

```plain
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
```


#### 整合输出格式说明到提示模板

通过解析器的`get_format_instructions()`方法生成格式约束说明，并将其嵌入提示模板，引导模型按指定结构输出：

```plain
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
```


#### 模型调用与输出解析

实例化模型，循环处理多组鲜花数据，通过解析器将模型输出转换为结构化字典，并整合为DataFrame：

```plain
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
```


#### 结果存储与验证

将整合后的DataFrame保存为CSV文件，并打印验证结果：

```plain
# 打印结构化数据（字典格式）
print("结构化输出结果：\n", df.to_dict(orient="records"))

# 保存为CSV文件（便于后续分析或系统集成）
df.to_csv("flowers_with_descriptions.csv", index=False)
```


### 输出解析的关键成果与技术价值

#### 典型输出结果

解析后的数据结构清晰，可直接被程序处理，示例如下：

```plain
[
    {
        "flower": "玫瑰",
        "price": "50",
        "description": "Unlock the secrets of rose beauty with our exquisite bouquet! 🌹💖 Hand-picked from the finest fields, these stems are bursting with charm and grace. From delicate pastels to bold, eye-catching hues, each bloom is a work of art in its own right. Indulge in their sweet fragrance and let yourself be transported to a world of tranquility and elegance. Treat yourself or a loved one to the ultimate rose experience! 💖🥳",
        "reason": "Based on customer feedback and market trends, we've found that customers are increasingly interested in unique and personalized flower arrangements. Our bouquet offers a range of colors and styles to choose from, allowing customers to create a customized gift that's tailored to their tastes and preferences. By emphasizing the hand-picked nature of our stems and the artistry involved in crafting each bouquet, we aim to appeal to customers who value quality, craftsmanship, and personalization. Additionally, the use of emojis and playful language helps to create a more approachable and friendly tone, which can help to build trust and rapport with our audience."
    },
    {
        "flower": "百合",
        "price": "30",
        "description": "Unlock the beauty of nature with our premium lily bouquet! 💐 Each stem is carefully curated to bring you a unique and mesmerizing display of colors, shapes, and textures. Whether it's for a special occasion or simply to brighten up your day, this stunning arrangement is sure to impress. 🎁",
        "reason": "Based on the product's high-end quality and unique feature of being able to customize the bouquet with different colors and styles, we want to highlight the exclusivity and personalization of the item in the description. The use of emojis and playful language adds a touch of friendliness and approachability to the copy, while still conveying the luxury and sophistication of the product."
    },
    {
        "flower": "康乃馨",
        "price": "20",
        "description": "🌼 Discover the sweet fragrance of Konigsberg's finest roses! Our carefully selected blooms are sure to brighten up any room with their vibrant colors and enchanting aroma. Treat yourself or gift them to someone special today! 💖",
        "reason": "We wanted to create a playful and inviting description that would appeal to customers looking for a unique and memorable gift experience. The use of emojis adds a fun and lighthearted touch, while the emphasis on the rose's fragrance helps to evoke a sense of nostalgia and romanticism. The inclusion of 'treat yourself or gift them' creates a sense of urgency and encourages customers to take action, making it more likely that they will make a purchase."
    }
]
```

生成的CSV文件结构如下（部分内容）：

| flower | price | description                                                  | reason                                          |
| ------ | ----- | ------------------------------------------------------------ | ----------------------------------------------- |
| 玫瑰   | 50    | Unlock the secrets of rose beauty with our exquisite bouquet... | Based on customer feedback and market trends... |


#### 核心技术价值

输出解析器在流程中发挥的关键作用可归纳为三点：  

- **规范输出结构**：通过`ResponseSchema`与格式说明，强制模型输出符合预期字段定义的内容，避免非结构化文本的歧义性；  
- **自动化解析转换**：无需人工干预，直接将模型输出转换为Python字典、DataFrame等结构化格式，大幅降低数据处理成本；  
- **支撑下游集成**：结构化数据可直接对接数据库存储、前端展示、业务逻辑判断等下游环节，实现“模型调用-数据处理-应用落地”的全流程自动化。  


通过输出解析机制，LangChain有效解决了“模型非结构化输出与程序结构化需求”的适配问题，为语言模型在实际应用中的规模化落地提供了关键技术支撑。