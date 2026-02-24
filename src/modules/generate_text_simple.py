# 这个模块包含了一个简单的文本生成函数 `generate_text_simple`，它使用一个语言模型来生成文本。
# 函数接受一个模型、一个初始的索引序列、要生成的新令牌数量以及上下文大小作为输入，并返回生成的索引序列。


import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # 每次生成一个新令牌，输入是当前的索引序列的最后 `context_size` 个令牌
        idx_cond = idx[:, -context_size:]
        # 使用模型预测下一个令牌的概率分布，取最后一个时间步的输出
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        # 从概率分布中选择概率最高的令牌作为下一个令牌
        idx_next = torch.argmax(logits, dim=-1, keepdim=True) 
        # 将新生成的令牌添加到索引序列中，准备下一轮生成
        idx = torch.cat((idx, idx_next), dim=1) 

    return idx