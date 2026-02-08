# Preprocess the raw data and print some statistics about it
# 可优化点：
# 1. 添加更多的文本清洗步骤，例如去除 HTML 标签、特殊字符等，以提高数据质量。
# 2. 使用更高级的分词工具（如 NLTK、spaCy）来进行分词，以获得更准确的结果。
# 3. 添加对文本进行小写化的步骤，以减少词汇表的大小。
# 4. 添加对停用词的处理，去除常见但无意义的词汇，以提高模型的性能。
# 5. 添加对词干提取或词形还原的步骤，以进一步减少词汇表的大小并提高模型的泛化能力。

import re
import read_raw_data

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', read_raw_data.raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break