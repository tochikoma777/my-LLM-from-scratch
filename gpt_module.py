

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_head_attention import MultiHeadAttention


GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
#初始化定义需要的各种超参数


class LayerNorm(nn.Module):
    #layer归一化的函数,可以避免信息泄露也可以稳定
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 #避免0的产生导致崩溃
        self.scale = nn.Parameter(torch.ones(emb_dim)) #动态的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim)) #动态的偏移参数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)#算平均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)#算方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)#归一化
        return self.scale * norm_x + self.shift #通过Ω和  œ 调整归一化后的值范围和位置


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            #这一步把它变得平滑了很多
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    #运行一次就线性两次激活一次
    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],               # 输入特征维度
            d_out=cfg["emb_dim"],              # 输出特征维度
            context_length=cfg["context_length"],  # 上下文长度
            num_heads=cfg["n_heads"],          # 注意力头的数量
            dropout=cfg["drop_rate"],          # Dropout 比例
            qkv_bias=cfg["qkv_bias"]           # 查询、键和值的偏置
        )  # 多头注意力模块，结合各种参数
        self.ff = FeedForward(cfg)  # 前馈神经网络模块
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 第一归一化层
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 第二归一化层
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # 残差连接的 Dropout

    def forward(self, x):
        # 对注意力模块的快捷连接
        shortcut = x
        x = self.norm1(x)  # 应用第一归一化层
        x = self.att(x)  # 通过多头注意力模块，形状为 [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)  # 应用 Dropout
        x = x + shortcut  # 将原始输入加回，实现残差连接

        # 对前馈网络模块的残差连接
        shortcut = x
        x = self.norm2(x)  # 应用第二归一化层
        x = self.ff(x)  # 通过前馈神经网络模块
        x = self.drop_shortcut(x)  # 应用 Dropout
        x = x + shortcut  # 将原始输入加回，实现残差连接

        return x


class GPTModel(nn.Module):#召唤GPT!
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        #新建字典、位置信息、还有dropout的比率设置
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        #解包操作

        self.trf_blocks = nn.Sequential(
        TransformerBlock(cfg),
        TransformerBlock(cfg),
        TransformerBlock(cfg)
                )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        #归一化
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        #输出头保证维度
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

# 测试 GPTModel 类的功能
tokenizer = tiktoken.get_encoding("gpt2")
#召唤gpt大神
batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
#编码输入文本
batch = torch.stack(batch, dim=0)
#按照横向来叠加两个向量
print(batch)


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)
#经典操作

total_params = sum(p.numel() for p in model.parameters())
#模型的总参数数量
print(f"Total number of parameters: {total_params:,}")


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # 预测单词的模块
    # idx 是当前上下文中的（batch, n_tokens）索引数组
    for _ in range(max_new_tokens):
        # 每次生成一个单词后，重新将其加入序列中
        # 如果当前上下文长度超过模型支持的最大上下文长度，则截取
        # 例如，如果LLM只支持5个token，而上下文长度为10
        # 那么只使用最后5个token作为上下文
        idx_cond = idx[:, -context_size:]
        # 如果idx的长度超过模型支持的上下文长度size，只保留最后size个token
        # 避免溢出
        # 获取预测结果
        with torch.no_grad():  # 在推理阶段，不需要计算梯度，因为没有反向传播
            # 这样可以减少存储开销
            logits = model(idx_cond)
            # 模型输出结果
        # 只关注最后一个时间步的输出
        # (batch, n_tokens, vocab_size) 变为 (batch, vocab_size)
        logits = logits[:, -1, :]
        # 关注最后一个时间步
        # 使用softmax函数计算概率
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        # 归一化
        # 获取具有最高概率值的词汇索引
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        # 获取概率最高的词汇索引
        # 将采样的索引添加到序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

start_context = "Hello, I am"
#模拟
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
#进行语义理解
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)
#最终输出格式


model.eval() # disable dropout
#在检验的时候不需要正则化了
out = generate_text_simple(
    model=model,
    #左边的参数名字,右边是函数传入的实际模型
    idx=encoded_tensor, #上下文的索引
    max_new_tokens=6, #最多运行六次,然后取结果概率最高的
    #初始文本➕6
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))
#输出长度还有每个单词的id

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)