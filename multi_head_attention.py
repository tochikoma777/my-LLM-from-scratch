# 说明：
# 1. 该代码示例展示了如何实现一个多头自注意力机制，使用 PyTorch 库来构建一个 MultiHeadAttention 类，该类包含了多头注意力的计算过程，包括线性变换、分头、缩放点积注意力、掩码处理和输出投影等步骤。
# 可优化点:
# 1. 可以添加异常处理来捕获可能的错误，例如在计算注意力分数时可能会发生的维度不匹配错误等。
# 2. 可以添加更多的功能，例如支持不同类型的注意力机制（如多头交叉注意力）、支持不同的输入格式（如批量输入）等，以提高模型的灵活性和适用性。


import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"
        #确保是可以被整除的
            

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim
        #初始化头的维度、数量
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        #头的输出结合线性层
        self.dropout = nn.Dropout(dropout)
        #进行dropout防止过拟合
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
        # 上三角掩码，确保因果性

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)
        #把输出的维度拆成头*头大小
        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        #转制维度,听说是为了更好的计算注意力
        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # 计算缩放点积注意力
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # 将掩码缩减到当前 token 数量，并转换为布尔型
        # 进而实现动态遮蔽,所以不用另开好几个数组
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 遮蔽矩阵
        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        #归一化
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        #头的合并
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        #对上下文向量的形状进行调整，确保输出的形状
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)