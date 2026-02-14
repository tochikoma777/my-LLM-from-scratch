# 说明：
# 1. 该代码示例展示了如何实现一个多头自注意力机制，使用 PyTorch 库来构建一个 MultiHeadAttention 类，该类包含了多头注意力的计算过程，包括线性变换、分头、缩放点积注意力、掩码处理和输出投影等步骤。
# 可优化点:
# 1. 可以添加异常处理来捕获可能的错误，例如在计算注意力分数时可能会发生的维度不匹配错误等。
# 2. 可以添加更多的功能，例如支持不同类型的注意力机制（如多头交叉注意力）、支持不同的输入格式（如批量输入）等，以提高模型的灵活性和适用性。



import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    # 输入维度、输出维度、上下文长度、dropout率、头的数量、是否使用偏置
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 确保输出维度可以被头的数量整除，以便每个头可以有相同的维度
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
            
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads 
        # 定义线性层来计算查询、键和值的表示，输入维度为 d_in，输出维度为 d_out，是否使用偏置由 qkv_bias 参数决定
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  
        self.dropout = nn.Dropout(dropout)
        # 注册一个上三角矩阵作为掩码，用于在计算注意力分数时遮蔽未来的时间步，确保模型只能关注当前和之前的时间步
        # register_buffer 方法用于注册一个持久化的缓冲区，这个缓冲区不会被视为模型的参数，但会随着模型一起保存和加载
        self.register_buffer( 
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # 计算查询、键和值的表示，输入 x 的形状为 (batch_size, num_tokens, d_in)，输出的 keys、queries 和 values 的形状为 (batch_size, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # 将 keys、queries 和 values 的形状调整为 (batch_size, num_tokens, num_heads, head_dim)，以便进行多头注意力的计算
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # 将 keys、queries 和 values 的维度进行转置，以便在计算注意力分数时进行矩阵乘法，新的形状为 (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # 计算注意力分数，使用缩放点积注意力的公式，attn_scores 的形状为 (batch_size, num_heads, num_tokens, num_tokens)，表示每个头对于每对查询和键的注意力分数
        attn_scores = queries @ keys.transpose(2, 3) 
        # 使用掩码来遮蔽未来的时间步，mask_bool 的形状为 (num_tokens, num_tokens)，其中上三角部分为 True，表示需要遮蔽的部分，attn_scores 中对应位置的分数将被设置为负无穷，以确保在计算 softmax 时这些位置的权重为零
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # 计算注意力权重，使用 softmax 函数对注意力分数进行归一化，并应用 dropout 以防止过拟合，attn_weights 的形状为 (batch_size, num_heads, num_tokens, num_tokens
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # 计算上下文向量，使用注意力权重对值进行加权求和，context_vec 的形状为 (batch_size, num_heads, num_tokens, head_dim)，然后将其转置回 (batch_size, num_tokens, num_heads, head_dim)，最后通过 contiguous 和 view 方法将其调整为 (batch_size, num_tokens, d_out)，并通过输出投影层进行线性变换得到最终的上下文向量
        context_vec = (attn_weights @ values).transpose(1, 2) 
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) 

        return context_vec

# 测试 MultiHeadAttention 类的功能
""" torch.manual_seed(123)

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], 
   [0.55, 0.87, 0.66], 
   [0.57, 0.85, 0.64], 
   [0.22, 0.58, 0.33], 
   [0.77, 0.25, 0.10], 
   [0.05, 0.80, 0.55]] 
)
batch = torch.stack((inputs, inputs), dim=0)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape) """