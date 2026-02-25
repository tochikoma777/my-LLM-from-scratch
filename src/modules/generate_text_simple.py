"""
简单文本生成模块 (Simple Text Generation Module)

本模块实现了基于贪婪解码 (Greedy Decoding) 的文本生成函数。
贪婪解码在每个时间步选择概率最高的token作为下一个输出，
实现简单但可能导致输出缺乏多样性。

核心概念：
- Auto-regressive Generation: 自回归生成，逐个token生成，每个新token依赖于之前生成的所有token
- Context Window: 上下文窗口，模型只能看到最近 context_size 个token
- Greedy Decoding: 贪婪解码，每步选择 logits 最大的token
"""

import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    使用贪婪解码生成文本序列 (Generate Text with Greedy Decoding)
    
    该函数实现了最基础的自回归文本生成算法：
    1. 将当前序列输入模型，获取下一个token的预测分布
    2. 选择概率最高的token（贪婪策略）
    3. 将该token添加到序列末尾
    4. 重复上述过程直到生成指定数量的token
    
    参数:
        model (nn.Module): 语言模型实例（如 GPTModel），必须实现 forward 方法
        idx (torch.Tensor): 初始token序列，形状为 (batch_size, seq_len)
                            通常包含提示词 (prompt) 的token IDs
        max_new_tokens (int): 需要生成的新token数量，不包括输入序列的长度
        context_size (int): 模型的最大上下文长度（位置编码的最大长度）
                            超过此长度的历史token会被截断
    
    返回:
        torch.Tensor: 生成的完整序列，形状为 (batch_size, seq_len + max_new_tokens)
    
    示例:
        >>> # 假设 tokenizer 将 "Hello" 编码为 [15496]
        >>> idx = torch.tensor([[15496]])  # 批次大小为1，序列长度为1
        >>> generated = generate_text_simple(model, idx, max_new_tokens=5, context_size=512)
        >>> print(generated.shape)  # torch.Size([1, 6])  (1个输入 + 5个新生成)
    
    注意事项:
        - 该函数使用 torch.no_grad() 上下文，禁用梯度计算，节省内存，适合推理
        - 贪婪解码的局限性：容易陷入重复循环，缺乏创造性，适合确定性任务
    """
    # 进入生成循环，迭代 max_new_tokens 次，每次生成一个新token
    for current_step in range(max_new_tokens):
        
        # 步骤 1: 裁剪上下文窗口
        # 模型只能处理最多 context_size 个token，如果序列太长则截断前面的部分
        # idx[:, -context_size:] 表示取每个批次样本的最后 context_size 个token
        # 形状变化: (batch_size, current_seq_len) -> (batch_size, min(current_seq_len, context_size))
        idx_cond = idx[:, -context_size:]
        
        # 步骤 2: 模型前向传播获取 logits
        # 使用 torch.no_grad() 禁用梯度计算，因为生成过程不需要反向传播
        # 这可以显著减少内存使用并加速计算
        with torch.no_grad():
            # 模型输出 logits，形状为 (batch_size, seq_len, vocab_size)
            # logits 表示每个位置每个token的未归一化分数
            logits = model(idx_cond)
        
        # 步骤 3: 提取最后一个时间步的预测
        # 我们只关心序列最后一个位置的下一个token预测
        # logits[:, -1, :] 的形状为 (batch_size, vocab_size)
        logits = logits[:, -1, :]
        
        # 步骤 4: 贪婪解码 - 选择概率最高的token
        # torch.argmax 返回指定维度上最大值的索引
        # dim=-1 表示在最后一个维度（vocab_size）上取 argmax
        # keepdim=True 保持维度，输出形状为 (batch_size, 1) 而非 (batch_size,)
        # idx_next 的形状: (batch_size, 1)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        # 步骤 5: 将新token追加到序列中
        # torch.cat 在指定维度上拼接张量
        # dim=1 表示在序列长度维度上拼接
        # 拼接后 idx 的形状增加1: (batch_size, current_seq_len) -> (batch_size, current_seq_len + 1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        # 可选：打印生成进度（调试用）
        # print(f"生成进度: {current_step + 1}/{max_new_tokens}, 新token ID: {idx_next.item()}")

    # 返回生成的完整序列，包括原始输入和所有新生成的token
    return idx