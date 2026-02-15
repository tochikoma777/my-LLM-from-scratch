# 说明：
# 该代码示例展示了如何使用 Python 的 re 模块进行文本预处理，包括分词、去除空白字符、构建词汇表等步骤。


import re
import read_raw_data

# 使用正则表达式进行分词，分隔符包括逗号、句号、冒号、分号、问号、感叹号、引号、括号、连字符和空白字符
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', read_raw_data.raw_text)
# 去除分词结果中的空字符串和仅包含空白字符的项     
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
# print(preprocessed[:30])
# 获取文本中所有唯一的词汇，并按字母顺序排序，构建词汇表
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)
# 构建一个词汇到整数的映射字典，使用 enumerate 函数为每个词汇分配一个唯一的整数 ID
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break