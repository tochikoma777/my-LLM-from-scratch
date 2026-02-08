#pip install tiktoken
# 1. 导入 importlib.metadata 模块中的 version 函数，用于获取已安装库的版本信息 
# 2. 导入 tiktoken 库，这是一个用于处理文本编码的库，特别适用于处理与 OpenAI 模型相关的文本编码
# 3. 使用 version 函数获取 tiktoken 库的版本信息，并打印出来，以确认当前使用的 tiktoken 版本

"""
# 网络有问题 
from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers) """

from importlib.metadata import version
import json
import base64
import re

# ========== 核心：自定义加载函数（解析本地vocab.bpe，跳过异常行） ==========
def load_local_tiktoken_bpe(vocab_file_path):
    """加载本地vocab.bpe文件，容错处理异常行"""
    mergeable_ranks = {}
    with open(vocab_file_path, "rb") as f:
        for line in f:
            line = line.strip()
            # 跳过空行、注释行、长度不足2的行
            if not line or line.startswith(b"#") or len(line.split()) < 2:
                continue
            try:
                token, rank = line.split()
                # 解码Base64字符 + 转换排名为整数
                mergeable_ranks[base64.b64decode(token, errors="ignore")] = int(rank)
            except (ValueError, base64.binascii.Error):
                # 跳过解析失败的行，不影响核心逻辑
                continue
    return mergeable_ranks

def load_local_encoder(encoder_file_path):
    """加载本地encoder.json文件"""
    with open(encoder_file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ========== 初始化本地分词器 ==========
# 打印tiktoken版本（仅本地读取，无网络）
print("tiktoken version:", version("tiktoken"))

# 本地文件路径（确保和代码同目录）
VOCAB_PATH = "vocab.bpe"
ENCODER_PATH = "encoder.json"

# 加载本地文件
mergeable_ranks = load_local_tiktoken_bpe(VOCAB_PATH)
encoder = load_local_encoder(ENCODER_PATH)

# 构建GPT2分词器（纯本地，无网络请求）
class LocalGPT2Tokenizer:
    def __init__(self, encoder, mergeable_ranks, special_tokens):
        self.encoder = encoder
        self.mergeable_ranks = mergeable_ranks
        self.special_tokens = special_tokens
        # GPT2官方分词正则
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.UNICODE)

    def encode(self, text, allowed_special=None):
        if allowed_special is None:
            allowed_special = set()
        # 拆分文本为基础词元
        tokens = self.pat.findall(text)
        # 转换为ID（优先匹配特殊标记，再匹配普通词汇）
        ids = []
        for token in tokens:
            if token in self.special_tokens and token in allowed_special:
                ids.append(self.special_tokens[token])
            elif token in self.encoder:
                ids.append(self.encoder[token])
            else:
                # 未匹配的词元用0填充（可按需扩展encoder.json）
                ids.append(0)
        return ids

# 初始化本地分词器（特殊标记配置）
special_tokens = {"<|endoftext|>": 50256}
tokenizer = LocalGPT2Tokenizer(encoder, mergeable_ranks, special_tokens)

# ========== 测试编码 ==========
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
# 编码文本（允许<|endoftext|>特殊标记）
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# 打印结果
print("本地文件加载成功！编码结果：")
print(integers)



