## 情人节玫瑰宣传语

情人节到啦，你的花店需要推销红色玫瑰，怎样通过LangChain来用程序的方式给咱们生成简短的宣传语呢？

很简单，你需要:
安装两个包

- 通过 `pip install langchain` 来安装LangChain

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

