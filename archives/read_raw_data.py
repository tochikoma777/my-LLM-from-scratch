# 说明：
# 该文件用于读取原始文本数据。它打开一个名为 "the-verdict.txt" 的文本文件，以 UTF-8 编码读取内容，并将其存储在变量 raw_text 中。然后，代码打印出文本的总字符数以及文本的前 100 个字符，以便检查数据是否正确加载。

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])