"""
GPT 模型架构模块 (GPT Model Architecture Module)

本模块实现了完整的 GPT (Generative Pre-trained Transformer) 模型架构，
包括以下核心组件：

1. MultiHeadAttention: 多头因果自注意力机制 (Causal Multi-Head Self-Attention)
2. LayerNorm: 层归一化 (Layer Normalization)
3. GELU: 高斯误差线性单元激活函数
4. FeedForward: 前馈神经网络
5. TransformerBlock: Transformer 基本构建块
6. GPTModel: 完整的 GPT 模型

架构特点：
- 仅使用 Decoder 部分（因果掩码确保自回归特性）
- 多头注意力机制捕捉不同子空间的信息
- 残差连接 (Residual Connections) 和层归一化稳定深层网络训练
- 位置编码 (Positional Embeddings) 注入序列顺序信息

参考: "Attention Is All You Need" (Vaswani et al., 2017)
      "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    多头因果自注意力机制 (Multi-Head Causal Self-Attention)
    
    这是 Transformer 架构的核心组件，允许模型在不同位置关注不同表示子空间的信息。
    "因果" (Causal) 意味着每个位置只能关注到当前及之前的位置，确保自回归特性。
    
    数学原理:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        
        其中:
        - Q (Query): 查询矩阵，表示"我要查什么信息"
        - K (Key): 键矩阵，表示"我有什么信息"
        - V (Value): 值矩阵，表示"信息的具体内容"
        - d_k: 每个头的维度，用于缩放防止 softmax 饱和
    
    多头机制:
        将 Q, K, V 投影到 h 个不同的子空间，并行计算注意力，最后拼接结果
    """
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        初始化多头注意力模块
        
        参数:
            d_in (int): 输入维度（嵌入维度）
            d_out (int): 输出维度，必须能被 num_heads 整除
            context_length (int): 最大序列长度，用于构建因果掩码
            dropout (float): Dropout 概率，用于防止过拟合
            num_heads (int): 注意力头的数量，决定并行注意力计算的数量
            qkv_bias (bool): 是否为 Q, K, V 的线性变换添加偏置项
        
        形状说明:
            输入 x: (batch_size, num_tokens, d_in)
            输出: (batch_size, num_tokens, d_out)
        """
        super().__init__()
        
        # 验证输出维度可被头数整除，确保每个头维度一致
        assert d_out % num_heads == 0, (
            f"输出维度 d_out ({d_out}) 必须能被注意力头数 num_heads ({num_heads}) 整除，"
            f"这样每个头的维度才是整数: {d_out} / {num_heads} = {d_out / num_heads}"
        )

        self.d_out = d_out              # 输出维度
        self.num_heads = num_heads      # 注意力头数
        self.head_dim = d_out // num_heads  # 每个头的维度（关键超参数）

        # 定义线性变换层，将输入投影到 Q, K, V 空间
        # 这些是可学习的参数，训练过程中不断优化
        # 尽管论文中 Q, K, V 通常维度相同，但这里允许 d_in != d_out 的灵活性
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 查询变换: x -> Q
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)    # 键变换: x -> K  
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 值变换: x -> V
        
        # 输出投影层，将多头注意力的拼接结果映射回 d_out 维度
        # 这是多头注意力机制的最后一步线性变换
        self.out_proj = nn.Linear(d_out, d_out)
        
        # Dropout 层，随机置零部分注意力权重，防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 注册因果掩码 (Causal Mask) 为模型缓冲区（非参数，但随模型保存）
        # torch.triu(..., diagonal=1) 生成上三角矩阵（对角线及以上为1，其余为0）
        # 形状: (context_length, context_length)
        # 作用: 在注意力计算中，将未来位置的注意力分数设为 -inf，实现因果性
        self.register_buffer(
            'mask', 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        # 掩码示例 (context_length=4):
        # [[0, 1, 1, 1],
        #  [0, 0, 1, 1],
        #  [0, 0, 0, 1],
        #  [0, 0, 0, 0]]
        # 1 的位置表示需要屏蔽（未来信息）

    def forward(self, x):
        """
        前向传播：计算多头因果自注意力
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, num_tokens, d_in)
        
        返回:
            torch.Tensor: 注意力输出，形状为 (batch_size, num_tokens, d_out)
        
        计算流程:
            1. 线性投影得到 Q, K, V
            2. 重塑为多头格式 (batch, heads, tokens, head_dim)
            3. 计算注意力分数: Q @ K^T
            4. 应用因果掩码屏蔽未来信息
            5. Softmax 归一化得到注意力权重
            6. 加权求和: weights @ V
            7. 合并多头结果并投影输出
        """
        # 获取输入维度信息
        batch_size, num_tokens, d_in = x.shape
        
        # 步骤 1: 线性投影计算 Q, K, V
        # 形状: (batch, tokens, d_in) -> (batch, tokens, d_out)
        keys = self.W_key(x)      # 键矩阵 K
        queries = self.W_query(x) # 查询矩阵 Q
        values = self.W_value(x)  # 值矩阵 V

        # 步骤 2: 重塑为多头格式
        # 目标形状: (batch, num_heads, num_tokens, head_dim)
        # 这样每个头可以独立计算注意力
        
        # view 操作将最后一个维度 d_out 拆分为 (num_heads, head_dim)
        # 例如: (2, 3, 768) -> (2, 3, 12, 64) 当 num_heads=12, head_dim=64
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # transpose 交换维度，将 num_heads 放到前面，便于批量矩阵乘法
        # 最终形状: (batch, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 步骤 3: 计算缩放点积注意力分数
        # 数学: scores = Q @ K^T / sqrt(d_k)
        # queries 形状: (batch, heads, tokens, head_dim)
        # keys.transpose(2, 3) 形状: (batch, heads, head_dim, tokens)
        # 结果形状: (batch, heads, tokens, tokens)
        # 即每个 token 对其他所有 token 的注意力分数
        attn_scores = queries @ keys.transpose(2, 3)
        
        # 步骤 4: 应用因果掩码 (Causal Masking)
        # 这是实现自回归特性的关键：每个位置只能看到当前及之前的位置
        
        # 根据当前序列长度裁剪掩码（处理变长序列）
        # bool() 转换为布尔类型，用于 masked_fill_
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        
        # masked_fill_: 将掩码为 True 的位置（未来位置）填充为 -inf
        # 这样 softmax 后这些位置的权重会变为 0
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # 步骤 5: 缩放和 Softmax 归一化
        # 缩放因子 sqrt(head_dim) 防止点积结果过大导致 softmax 梯度消失
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        # 应用 Dropout 到注意力权重（训练时随机丢弃部分连接）
        attn_weights = self.dropout(attn_weights)
        
        # 步骤 6: 计算上下文向量（注意力加权求和）
        # attn_weights: (batch, heads, tokens, tokens)
        # values: (batch, heads, tokens, head_dim)
        # 结果: (batch, heads, tokens, head_dim)
        context_vec = (attn_weights @ values)
        
        # 转置回 (batch, tokens, heads, head_dim) 以便合并多头
        context_vec = context_vec.transpose(1, 2)

        # 步骤 7: 合并多头并投影输出
        # contiguous() 确保内存连续，view 操作需要连续的内存布局
        # 合并多头: (batch, tokens, heads, head_dim) -> (batch, tokens, d_out)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        
        # 最终线性投影，允许模型学习如何组合不同头的信息
        context_vec = self.out_proj(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    """
    层归一化模块 (Layer Normalization)
    
    对输入张量的最后一个维度（特征维度）进行归一化，稳定深层网络训练。
    与 BatchNorm 不同，LayerNorm 对每个样本独立计算统计量，不依赖批次大小，
    因此更适合序列建模和变长输入。
    
    数学公式:
        y = scale * (x - mean) / sqrt(var + eps) + shift
        
        其中:
        - mean, var: 在特征维度上计算的均值和方差
        - eps: 防止除零的小常数
        - scale, shift: 可学习的仿射变换参数（gamma 和 beta）
    """
    
    def __init__(self, emb_dim):
        """
        初始化层归一化模块
        
        参数:
            emb_dim (int): 嵌入维度（特征维度），归一化操作在这个维度上进行
        """
        super().__init__()
        
        # eps (epsilon): 防止除以零的极小常数，保证数值稳定性
        # 默认值 1e-5 是深度学习中常用的经验值
        self.eps = 1e-5
        
        # scale (gamma): 可学习的缩放参数，初始化为全1
        # 允许模型在需要时恢复原始分布的尺度
        self.scale = nn.Parameter(torch.ones(emb_dim))
        
        # shift (beta): 可学习的偏移参数，初始化为全0
        # 允许模型在需要时恢复原始分布的位置
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """
        前向传播：应用层归一化
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (..., emb_dim)
        
        返回:
            torch.Tensor: 归一化后的张量，形状与输入相同
        """
        # 在最后一个维度（特征维度）上计算均值
        # keepdim=True 保持维度，便于广播操作
        # 例如: (2, 3, 768) -> (2, 3, 1)
        mean = x.mean(dim=-1, keepdim=True)
        
        # 计算方差（无偏估计关闭，使用总体方差）
        # unbiased=False: 除以 N 而非 N-1，与原始 Transformer 论文一致
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 归一化: (x - mean) / sqrt(var + eps)
        # eps 确保数值稳定性，防止方差为0时除零错误
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        # 应用可学习的仿射变换: scale * norm_x + shift
        # 这允许模型在训练过程中学习最优的分布尺度和平移
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    GELU 激活函数模块 (Gaussian Error Linear Unit)
    
    GELU 是一种平滑的非线性激活函数，相比 ReLU 具有更好的梯度特性。
    它在 Transformer 架构中广泛使用（如 BERT、GPT）。
    
    数学定义:
        GELU(x) = x * Φ(x)
        其中 Φ(x) 是标准正态分布的累积分布函数 (CDF)
    
    近似公式（Hendrycks & Gimpel, 2016）:
        GELU(x) ≈ 0.5 * x * (1 + tanh[sqrt(2/π) * (x + 0.044715 * x^3)])
    
    特性:
        - 平滑可导，处处非零梯度
        - 对负数输入有非零输出（不像 ReLU 直接截断为0）
        - 近似模拟随机正则化的效果
    """
    
    def __init__(self):
        """初始化 GELU 激活函数（无参数）"""
        super().__init__()

    def forward(self, x):
        """
        应用 GELU 激活函数
        
        参数:
            x (torch.Tensor): 输入张量
        
        返回:
            torch.Tensor: 经过 GELU 变换后的张量
        """
        # 使用 tanh 近似计算 GELU
        # sqrt(2/pi) 是归一化常数
        # 0.044715 是近似系数，使误差最小
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    前馈神经网络模块 (Feed-Forward Network)
    
    每个 Transformer 块中的前馈子层，对每个位置独立应用相同的非线性变换。
    结构: Linear -> GELU -> Linear
    
    扩展-收缩结构:
        - 第一个线性层将维度从 emb_dim 扩展到 4*emb_dim
        - 第二个线性层将维度从 4*emb_dim 收缩回 emb_dim
        这种"瓶颈"结构增加了模型的非线性表达能力
    
    参考: "Attention Is All You Need" 中 d_ff = 4 * d_model
    """
    
    def __init__(self, cfg):
        """
        初始化前馈网络
        
        参数:
            cfg (dict): 配置字典，必须包含:
                - emb_dim: 嵌入维度（输入/输出维度）
        """
        super().__init__()
        
        # 使用 Sequential 容器组织网络层
        self.layers = nn.Sequential(
            # 第一层: 扩展维度，增加非线性表达能力
            # 输入: (batch, tokens, emb_dim) -> 输出: (batch, tokens, 4*emb_dim)
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            
            # GELU 非线性激活
            GELU(),
            
            # 第二层: 投影回原始维度
            # 输入: (batch, tokens, 4*emb_dim) -> 输出: (batch, tokens, emb_dim)
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch, tokens, emb_dim)
        
        返回:
            torch.Tensor: 输出张量，形状与输入相同
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    Transformer 基本构建块 (Transformer Block)
    
    这是 GPT 模型的核心组件，由以下子层组成：
    1. 层归一化 (LayerNorm)
    2. 多头因果自注意力 (Multi-Head Causal Self-Attention)
    3. 残差连接 (Residual Connection)
    4. 层归一化 (LayerNorm)
    5. 前馈网络 (Feed-Forward Network)
    6. 残差连接 (Residual Connection)
    
    注意: 这里使用 Pre-LN 结构（注意力/FFN 之前归一化），
          与原始 Transformer 的 Post-LN 不同，训练更稳定
    """
    
    def __init__(self, cfg):
        """
        初始化 Transformer 块
        
        参数:
            cfg (dict): 配置字典，包含:
                - emb_dim: 嵌入维度
                - context_length: 上下文长度
                - n_heads: 注意力头数
                - drop_rate: Dropout 概率
                - qkv_bias: 是否使用 QKV 偏置
        """
        super().__init__()
        
        # 多头自注意力子层
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],           # 输入维度
            d_out=cfg["emb_dim"],          # 输出维度
            context_length=cfg["context_length"],  # 上下文长度（用于掩码）
            num_heads=cfg["n_heads"],      # 注意力头数
            dropout=cfg["drop_rate"],      # Dropout 概率
            qkv_bias=cfg["qkv_bias"]       # QKV 偏置设置
        )
        
        # 前馈网络子层
        self.ff = FeedForward(cfg)
        
        # 第一个层归一化（注意力之前）
        self.norm1 = LayerNorm(cfg["emb_dim"])
        
        # 第二个层归一化（前馈网络之前）
        self.norm2 = LayerNorm(cfg["emb_dim"])
        
        # 用于残差连接的 Dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        前向传播：应用 Transformer 块
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch, tokens, emb_dim)
        
        返回:
            torch.Tensor: 输出张量，形状与输入相同
        
        计算流程:
            # 注意力子层（带残差连接）
            shortcut = x
            x = LayerNorm(x)
            x = Attention(x)
            x = Dropout(x)
            x = x + shortcut  # 残差连接
            
            # 前馈子层（带残差连接）
            shortcut = x
            x = LayerNorm(x)
            x = FFN(x)
            x = Dropout(x)
            x = x + shortcut  # 残差连接
        """
        # ===== 注意力子层（带残差连接）=====
        # 保存输入用于残差连接
        shortcut = x
        
        # Pre-LN: 在注意力之前应用层归一化
        x = self.norm1(x)
        
        # 计算多头自注意力
        x = self.att(x)
        
        # 应用 Dropout（正则化）
        x = self.drop_shortcut(x)
        
        # 残差连接: 输出 = 注意力输出 + 原始输入
        # 残差连接帮助梯度流动，缓解梯度消失问题，支持深层网络训练
        x = x + shortcut

        # ===== 前馈子层（带残差连接）=====
        # 保存当前输入用于残差连接
        shortcut = x
        
        # Pre-LN: 在前馈网络之前应用层归一化
        x = self.norm2(x)
        
        # 应用前馈网络
        x = self.ff(x)
        
        # 应用 Dropout
        x = self.drop_shortcut(x)
        
        # 残差连接: 输出 = 前馈输出 + 输入
        x = x + shortcut
        
        return x


class GPTModel(nn.Module):
    """
    GPT 模型主类 (Generative Pre-trained Transformer)
    
    完整的 GPT 架构，包含:
    1. 词嵌入层 (Token Embeddings): 将离散 token ID 映射为连续向量
    2. 位置编码层 (Positional Embeddings): 注入位置信息
    3. Dropout 层: 正则化
    4. N 个堆叠的 Transformer 块: 核心特征提取
    5. 最终层归一化: 稳定输出
    6. 输出线性层: 将特征映射到词汇表维度，得到 logits
    
    架构特点:
        - 纯解码器结构（Decoder-only）
        - 因果掩码确保自回归生成
        - 可扩展的层数和维度配置
    """
    
    def __init__(self, cfg):
        """
        初始化 GPT 模型
        
        参数:
            cfg (dict): 模型配置字典，包含:
                - vocab_size: 词汇表大小（决定输出维度）
                - context_length: 最大序列长度（上下文窗口）
                - emb_dim: 嵌入维度（模型宽度）
                - n_heads: 注意力头数
                - n_layers: Transformer 块数量（模型深度）
                - drop_rate: Dropout 概率
                - qkv_bias: QKV 线性层是否使用偏置
        """
        super().__init__()
        
        # ===== 输入嵌入层 =====
        # 词嵌入层: 将 token ID (整数) 映射为稠密向量
        # 权重矩阵形状: (vocab_size, emb_dim)
        # 输入: (batch, seq_len) 的整数索引
        # 输出: (batch, seq_len, emb_dim) 的稠密向量
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        
        # 位置编码层: 将位置索引 (0, 1, 2, ..., context_length-1) 映射为向量
        # 权重矩阵形状: (context_length, emb_dim)
        # 作用: 为模型提供序列顺序信息（Transformer 本身不具备位置感知）
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        # Dropout 层: 随机丢弃部分嵌入元素，防止过拟合
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # ===== Transformer 块堆叠 =====
        # 使用 Sequential 容器堆叠 n_layers 个 Transformer 块
        # 每个块包含: 多头注意力 + 前馈网络 + 残差连接 + 层归一化
        # 列表推导式创建多个独立的 TransformerBlock 实例
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # ===== 输出层 =====
        # 最终层归一化: 在输出前应用，稳定训练
        self.final_norm = LayerNorm(cfg["emb_dim"])
        
        # 输出投影层: 将嵌入维度映射回词汇表维度
        # 形状: (batch, seq_len, emb_dim) -> (batch, seq_len, vocab_size)
        # 输出即为每个位置每个 token 的 logits（未归一化分数）
        # bias=False: 原始 GPT-2 实现中输出层不使用偏置
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        前向传播：计算输入序列的 logits
        
        参数:
            in_idx (torch.Tensor): 输入 token IDs，形状为 (batch_size, seq_len)
        
        返回:
            torch.Tensor: 输出 logits，形状为 (batch_size, seq_len, vocab_size)
                         每个位置包含词汇表上每个 token 的预测分数
        
        计算流程:
            1. 词嵌入: token ID -> 向量
            2. 位置编码: 位置 ID -> 向量，与词嵌入相加
            3. Dropout 正则化
            4. 通过 N 层 Transformer 块提取特征
            5. 最终层归一化
            6. 投影到词汇表维度得到 logits
        """
        # 获取批次大小和序列长度
        batch_size, seq_len = in_idx.shape
        
        # 步骤 1: 词嵌入查找
        # tok_embeds 形状: (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        
        # 步骤 2: 位置编码
        # 生成位置索引: [0, 1, 2, ..., seq_len-1]
        # 自动放置到与输入相同的设备（CPU/GPU）
        pos_indices = torch.arange(seq_len, device=in_idx.device)
        
        # 位置编码查找，形状: (seq_len, emb_dim)
        pos_embeds = self.pos_emb(pos_indices)
        
        # 词嵌入与位置编码相加，融合语义信息和位置信息
        # 广播机制: pos_embeds 自动扩展为 (batch_size, seq_len, emb_dim)
        x = tok_embeds + pos_embeds
        
        # 步骤 3: 应用 Dropout
        x = self.drop_emb(x)
        
        # 步骤 4: 通过 Transformer 块堆叠
        # 每个块提取更高级的特征表示
        x = self.trf_blocks(x)
        
        # 步骤 5: 最终层归一化
        x = self.final_norm(x)
        
        # 步骤 6: 投影到词汇表维度
        # logits 形状: (batch_size, seq_len, vocab_size)
        # 可用于计算交叉熵损失或生成下一个 token
        logits = self.out_head(x)
        
        return logits