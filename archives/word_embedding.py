# 说明:
# 1. 该代码示例展示了如何使用 tiktoken 库来编码文本，特别是针对 GPT-2 模型的编码器  
# 可优化点:
# 1. 可以添加异常处理来捕获可能的错误，例如在获取版本信息或编码文本时可能会发生的错误



#pip install tiktoken
# 导入 importlib.metadata 模块中的 version 函数，用于获取已安装库的版本信息
from importlib.metadata import version 
# 导入 tiktoken 库，这是一个用于处理文本编码的库，特别适用于处理与 OpenAI 模型相关的文本编码
import tiktoken

print("tiktoken version:", version("tiktoken"))
# 获取 GPT-2 模型使用的编码器，这个编码器将文本转换为整数 ID 的形式，适用于 GPT-2 模型的输入
tokenizer = tiktoken.get_encoding("gpt2") 

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
# 使用 tokenizer 的 encode 方法将文本转换为整数 ID 的列表，allowed_special 参数指定允许的特殊标记，这里允许 "<|endoftext|>" 作为特殊标记
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers) 





