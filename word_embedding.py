#pip install tiktoken
# 1. 导入 importlib.metadata 模块中的 version 函数，用于获取已安装库的版本信息 
# 2. 导入 tiktoken 库，这是一个用于处理文本编码的库，特别适用于处理与 OpenAI 模型相关的文本编码
# 3. 使用 version 函数获取 tiktoken 库的版本信息，并打印出来，以确认当前使用的 tiktoken 版本

from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers) 





