# 说明：
# 1. 该代码实现了一个基于 Transformer 架构的 GPT 模型，包含了多头注意力机制、前馈神经网络、层归一化等组件。
# 2. 模型的配置参数（如词汇表大小、上下文长度、嵌入维度、注意力头数、层数、Dropout 比例等）被定义在 GPT_CONFIG_124M 字典中。
# 3. 模型的前向传播过程包括了输入的嵌入、位置编码、多层 Transformer 块的处理、最终的归一化和输出头的线性变换。
# 可优化点：
# 1. 可以添加更多的注释来解释每个组件的功能和实现细节，以提高代码的可读性。
# 2. 可以添加更多的测试用例来验证模型的功能和性能，例如测试不同输入文本的生成结果，或者比较不同配置下模型的输出差异。
# 3. 可以添加训练代码来训练模型，并评估其在文本生成任务上的性能，例如使用交叉熵损失函数和优化器来更新模型参数，并在验证集上评估生成文本的质量。



# 导入所需的库和模块
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_head_attention import MultiHeadAttention
from random_sampling_outout import generate

# 定义 GPT 模型的配置参数，包括词汇表大小、上下文长度、嵌入维度、注意力头数、层数、Dropout 比例和查询-键-值偏置等
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

# 定义层归一化类 LayerNorm，用于对输入张量进行归一化处理
class LayerNorm(nn.Module):

    # 初始化 LayerNorm 类，接受嵌入维度作为参数，并定义可学习的缩放参数和偏移参数
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) 

    # 运行前向传播，计算输入张量的均值和方差，并应用归一化、缩放和偏移操作
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# 定义 GELU 激活函数类，继承自 nn.Module，并实现前向传播方法，使用 GELU 函数对输入张量进行非线性变换
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # GELU 函数的近似公式为：0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


# 定义前馈神经网络类 FeedForward，包含两个线性层和一个 GELU 激活函数，用于在 Transformer 块中对输入进行非线性变换
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


# 定义 Transformer 块类 TransformerBlock，包含一个多头注意力模块、一个前馈神经网络模块、两个层归一化层和一个残差连接的 Dropout 模块，用于在 GPT 模型中构建多个 Transformer 块
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],                   # 输入特征维度
            d_out=cfg["emb_dim"],                  # 输出特征维度
            context_length=cfg["context_length"],  # 上下文长度
            num_heads=cfg["n_heads"],              # 注意力头的数量
            dropout=cfg["drop_rate"],              # Dropout 比例
            qkv_bias=cfg["qkv_bias"]               # 查询、键和值的偏置
        ) 
        self.ff = FeedForward(cfg) 
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"]) 
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        
        # 对多头注意力模块的残差连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x) 
        x = self.drop_shortcut(x) 
        x = x + shortcut

        # 对前馈网络模块的残差连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x) 
        x = self.drop_shortcut(x) 
        x = x + shortcut

        return x



# 定义 GPT 模型类 GPTModel，包含输入嵌入层、位置嵌入层、多个 Transformer 块、最终的层归一化和输出头，用于实现基于 Transformer 架构的 GPT 模型
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 定义输入嵌入层和位置嵌入层，输入嵌入层将词汇索引转换为嵌入向量，位置嵌入层为每个位置生成一个嵌入向量，以便模型能够捕捉位置信息
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 定义多个 Transformer 块，使用 nn.Sequential 来堆叠多个 TransformerBlock 实例，数量由 cfg["n_layers"] 决定，解包操作将列表中的每个 TransformerBlock 实例作为单独的参数传递给 nn.Sequential
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        # 定义最终的层归一化和输出头，层归一化用于对 Transformer 块的输出进行归一化处理，输出头是一个线性层，将嵌入维度转换为词汇表大小，以便生成文本时能够预测每个位置的下一个词汇的概率分布
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    # 定义前向传播方法，接受输入索引张量 in_idx，计算输入的嵌入和位置嵌入，将它们相加后通过 Dropout 进行正则化，然后依次通过多个 Transformer 块进行处理，最后通过层归一化和输出头得到预测的词汇分布 logits，并返回 logits
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
    

# 测试 GPTModel 类的功能，首先使用 tiktoken 库获取 GPT-2 模型的编码器，然后将两个文本输入编码为整数索引，并将它们堆叠成一个批次输入到模型中，最后打印输入批次和模型输出的形状以及输出内容，并计算模型的总参数数量
""" tokenizer = tiktoken.get_encoding("gpt2")
batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

# 创建 GPTModel 实例，并将批次输入传递给模型进行前向传播，打印输出的形状和内容，以及模型的总参数数量
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print(out)
print(out.shape)
# 输出的形状为 (batch_size, seq_len, vocab_size)，其中 batch_size 是输入批次的大小，seq_len 是输入序列的长度，vocab_size 是词汇表的大小
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}") """


# 定义一个简单的文本生成函数 generate_text_simple，接受模型、输入索引、最大新令牌数量和上下文大小作为参数，在每次迭代中使用模型预测下一个词汇的概率分布，并选择概率最高的词汇作为下一个输入，最终返回生成的文本索引序列
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        # 在生成过程中不需要计算梯度，因此使用 torch.no_grad() 上下文管理器来禁用梯度计算，以节省内存和提高性能
        with torch.no_grad(): 
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)  
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # print("idx_next:", idx_next)
        # print("idx:", idx) 
        idx = torch.cat((idx, idx_next), dim=1)
        # print("new idx:", idx)  

    return idx


# 测试文本生成函数 generate_text_simple，首先定义一个初始文本并将其编码为整数索引，然后将编码后的索引转换为张量并添加批次维度，最后调用 generate_text_simple 函数生成新的文本索引序列，并使用 tokenizer 将生成的索引序列解码回文本形式进行输出
""" start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)


# 将模型设置为评估模式，以禁用 Dropout 和其他训练特定的行为，然后调用 generate_text_simple 函数生成新的文本索引序列，并打印输出的索引序列和长度，以及解码后的文本内容
model.eval() 
out = generate_text_simple(
    model=model,
    idx=encoded_tensor, 
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text) """




def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")

model = GPTModel(GPT_CONFIG_124M)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))



