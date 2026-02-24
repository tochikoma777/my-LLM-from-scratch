# 这个模块定义了GPT模型的核心组件，包括多头自注意力机制、前馈网络、层归一化以及整个Transformer块和GPT模型的结构。
# 这些组件是构建GPT模型的基础，负责处理输入数据并生成输出文本的概率分布。


import torch
import torch.nn as nn


# ==============================================================================
# 【GPT 整体结构】
# ┌─────────────────────────────────────────────────────────────────────┐
# │                        输入层 Input Layer                            │
# │  ┌───────────────┐    ┌─────────────────┐    ┌────────────────────┐ │
# │  │Token Embedding│    │Position Encoding│    │ Add + Layer Norm   │ │
# │  │ （词嵌入）     │    │ （位置编码）     │    │ （相加+层归一化）   │  │
# │  └───────────────┘    └─────────────────┘    └────────────────────┘ │
# └───────────────────────────────────┬─────────────────────────────────┘
#                                     │
# ┌───────────────────────────────────▼─────────────────────────────────┐
# │                    Transformer Decoder Stack（N层堆叠）             │
# │  ┌─────────────────────────────────────────────────────────────┐    │
# │  │                Decoder Block（单块解码器）                   │    │
# │  │  ┌───────────────┐    ┌───────────────┐    ┌──────────────┐ │    │
# │  │  │ Layer Norm    │──▶│ Multi-Head    │──▶ │ Add（残差）   │ │    │
# │  │  │ （层归一化）   │    │ Attention     │    │              │ │    │
# │  │  └───────────────┘    │ （多头自注意力）│    └──────────────┘ │    │
# │  │                       └───────────────┘            │         │    │
# │  │                                                   ▼          │    │
# │  │  ┌───────────────┐    ┌───────────────┐    ┌──────────────┐  │    │
# │  │  │ Layer Norm    │──▶ │ Feed Forward │──▶ │ Add（残差）   │  │    │
# │  │  │ （层归一化）   │    │ Network       │    │              │  │    │
# │  │  └───────────────┘    │ （前馈网络）   │    └──────────────┘  │    │
# │  │                       └───────────────┘                      │    │
# │  └─────────────────────────────────────────────────────────────┘    │
# │  ┌─────────────────────────────────────────────────────────────┐    │
# │  │                Decoder Block（第2层）                        │    │
# │  └─────────────────────────────────────────────────────────────┘    │
# │  ┌─────────────────────────────────────────────────────────────┐    │
# │  │                ......（共N层Decoder Block）                  │    │
# │  └─────────────────────────────────────────────────────────────┘    │
# └───────────────────────────────────┬─────────────────────────────────┘
#                                     │
# ┌───────────────────────────────────▼─────────────────────────────────┐
# │                        输出层 Output Layer                          │
# │  ┌───────────────┐    ┌────────────────┐    ┌────────────────────┐  │
# │  │ Final Layer   │    │ Linear Layer   │    │ Softmax            │  │
# │  │ Norm          │──▶│ （线性层）      │──▶│ （输出token概率）   │  │
# │  │ （最终层归一化 │    │                │    │                    │  │
# │  └───────────────┘    └────────────────┘    └────────────────────┘  │
# └─────────────────────────────────────────────────────────────────────┘
# 
# ==============================================================================


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # 缩放点积注意力的线性变换层，分别用于查询、键和值的计算
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 输出线性变换层，用于将多头注意力的输出映射回输入维度
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # causal mask 用于确保模型只能关注当前和之前的token，防止信息泄露
        # register_buffer 用于注册一个持久的缓冲区，这个缓冲区不会被视为模型参数，但会随模型一起保存和加载
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        # 线性变换计算查询、键和值，并重塑为多头格式
        keys = self.W_key(x)  
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力分数，形状为 (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  
        # 应用 causal mask，确保模型只能关注当前和之前的token
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # 对注意力分数进行缩放和softmax归一化，得到注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 将多头注意力的输出重新组合并通过输出线性变换层映射回输入维度
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) 

        return context_vec


# 层归一化模块，用于对输入进行归一化处理，稳定训练过程
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# GELU激活函数，常用于Transformer模型中，提供非线性变换
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # GELU函数的数学表达式为：0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# 前馈网络模块，包含两个线性层和一个GELU激活函数，用于对Transformer块中的每个位置进行非线性变换
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# Transformer块模块，包含一个多头自注意力层和一个前馈网络层，以及相应的层归一化和残差连接
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 残差连接 for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   
        x = self.drop_shortcut(x)
        x = x + shortcut  

        # 残差连接 for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


# GPT模型模块，包含输入层（词嵌入和位置编码）、N层堆叠的Transformer块，以及输出层（最终层归一化和线性层）
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds 
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits