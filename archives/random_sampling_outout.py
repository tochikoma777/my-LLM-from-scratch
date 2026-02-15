# 说明：
# 该文件定义了一个名为 generate 的函数，用于使用 GPT 模型生成文本。函数接受模型、输入索引、最大新令牌数、上下文大小、温度、top_k 和 eos_id 等参数。在生成过程中，函数通过循环获取模型的输出 logits，并根据指定的采样策略（如 top_k 采样和温度校正）选择下一个 token 的索引。生成过程会在遇到指定的结束标记（eos_id）时提前停止。最后，函数返回生成的索引序列。


import torch
from torch.utils.data import Dataset, DataLoader


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        #计算预测值,但是切最后一个
        # New: Filter logits with top_k sampling
        #top K采样
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
            
        # New: Apply temperature scaling
        #温度校正
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            #从概率分布中采样下一个 token 

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
            #如果未启用采样，选择概率最高的 token 作为下一个 token 
        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx
