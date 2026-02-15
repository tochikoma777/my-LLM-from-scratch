# 说明：
# 该代码示例展示了如何实现一个简单的 GPT 模型预训练模块，包括下载预训练的 GPT-2 模型权重、定义 GPT 模型结构、加载预训练权重到模型中，以及使用该模型进行文本生成和评估等步骤。
# 主要目的是将预训练的模型参数加载到一个 GPT 模型中，以便在后续的训练过程中使用这些预训练的权重来加速模型的收敛和提高性能。


import os
import urllib.request
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from gpt_module import generate_text_simple
from gpt_module import GPTModel
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# 定义 GPT 模型的基本配置，包括词汇表大小、上下文长度、嵌入维度、注意力头数、层数、丢弃率和是否使用 qkv_bias 等参数。这些配置将作为 GPT 模型的基础设置，后续会根据具体的预训练模型大小进行更新。
GPT_CONFIG_124M = {
    "vocab_size": 50257,   
    "context_length": 256, 
    "emb_dim": 768,        
    "n_heads": 12,         
    "n_layers": 12,        
    "drop_rate": 0.1,      
    "qkv_bias": False      
}

tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

# 定义一个 GPTDatasetV1 类，继承自 PyTorch 的 Dataset 类，用于将文本数据转换为模型训练所需的输入和目标格式。该类在初始化时接受文本数据、分词器、最大长度和步长等参数，并将文本数据分割成适合模型输入的块，同时生成对应的目标块。
class GPTDatasetV1(Dataset): 
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# 定义一个函数 create_dataloader_v1，用于创建一个 DataLoader 对象，该对象使用 GPTDatasetV1 类来加载文本数据，并根据指定的批次大小、最大长度、步长等参数进行数据加载和预处理，以便在模型训练过程中使用。
def create_dataloader_v1(txt, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

# 下载文本数据并进行预处理，首先检查本地是否已经存在指定的文本文件，如果不存在则从指定的 URL 下载文本数据并保存到本地文件中，如果文件已经存在则直接从文件中加载文本数据。然后打印文本数据的前 99 个字符和后 99 个字符，以及文本的总字符数和总令牌数，以便了解文本数据的基本情况。
file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
    print(f"File downloaded successfully and saved as '{file_path}'")
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    print(f"File '{file_path}' already exists. Loaded content from file.")

# 打印文本数据的前 99 个字符和后 99 个字符，以及文本的总字符数和总令牌数，以便了解文本数据的基本情况
""" print(text_data[:99])
print(text_data[-99:])
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens) """


train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)
# 创建训练和验证数据加载器，使用 create_dataloader_v1 函数将训练数据和验证数据分别转换为 DataLoader 对象，以便在模型训练过程中使用。这些 DataLoader 对象将根据指定的批次大小、最大长度、步长等参数进行数据加载和预处理。
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

total_tokens = len(tokenizer.encode(text_data))
if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")

# print(f"Train loader:{len(train_loader)}")

# 计算一个批次的损失函数，首先将输入批次和目标批次转移到指定的设备上，然后将输入批次传递给模型进行前向传播，得到模型的输出 logits。接着使用交叉熵损失函数计算 logits 和目标批次之间的损失，并返回该损失值。
def calc_loss_batch(input_batch, target_batch, model, device): 
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss
# 计算整个数据加载器的平均损失函数，首先将模型设置为评估模式，然后迭代数据加载器中的每个批次，使用 calc_loss_batch 函数计算每个批次的损失，并累加到 total_loss 中。最后根据指定的 num_batches 参数计算平均损失并返回，如果 num_batches 为 None 则使用数据加载器中的所有批次进行计算。
def calc_loss_loader(data_loader, model, device, num_batches=None): 
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("********************************************")
print("Using device:", device)
print("********************************************")

""" model.to(device)

torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss) """

# 定义一个函数 train_model_simple，用于训练 GPT 模型，接受模型、训练数据加载器、验证数据加载器、优化器、设备、训练轮数、评估频率、评估迭代次数、起始上下文和分词器等参数。在每个训练轮中，模型将迭代训练数据加载器中的每个批次，计算损失并更新模型权重。根据指定的评估频率，模型还会在训练过程中定期评估训练损失和验证损失，并打印相关信息。最后返回训练损失、验证损失和已处理的令牌数量等信息。
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    #初始化训练模型而且给了空的队列
    # Main training loop
    for epoch in range(num_epochs):#训练次数
        model.train()  # Set model to training mode
        #转移到训练模块
        for input_batch, target_batch in train_loader:
            #从loader里面调出输入跟目标
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            #清空所有函数的梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            #计算损失函数
            loss.backward() # Calculate loss gradients
            #反向传播优化
            optimizer.step() # Update model weights using loss gradients
            #更新权重
            tokens_seen += input_batch.numel()
            #加一下一共有多少
            global_step += 1
            #看一下一共训练了多少步
            # Optional evaluation step
            if global_step % eval_freq == 0:
                #按照一定的步数进行记录
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                #计算损失函数
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                #加到list中
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

# 定义一个函数 evaluate_model，用于评估 GPT 模型的性能，接受模型、训练数据加载器、验证数据加载器、设备和评估迭代次数等参数。该函数首先将模型设置为评估模式，然后使用 torch.no_grad() 上下文管理器禁用梯度计算，以节省内存和提高性能。接着分别计算训练损失和验证损失，并返回这两个损失值。最后在评估结束后切换回训练模式，确保模型能继续用于训练。
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    #评价模块
    model.eval()
    #检验模式
    with torch.no_grad():
        #我认为的双保险,防止梯度更新
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    #	在评估结束后切换回训练模式，确保模型能继续用于训练。
    return train_loss, val_loss

# 定义一个函数 generate_and_print_sample，用于使用 GPT 模型生成文本样本并打印输出。该函数接受模型、分词器、设备和起始上下文等参数，首先将模型设置为评估模式，然后使用 text_to_token_ids 函数将起始上下文编码为整数索引，并将其转移到指定的设备上。接着使用 torch.no_grad() 上下文管理器禁用梯度计算，并调用 generate_text_simple 函数生成新的文本索引序列。最后使用 token_ids_to_text 函数将生成的索引序列解码回文本形式，并打印输出。
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# 定义一个函数 load_weights_into_gpt，用于将预训练的 GPT 模型权重加载到一个 GPTModel 实例中。该函数接受一个 GPTModel 实例和一个包含预训练权重的参数字典作为输入，然后逐层将预训练权重赋值给 GPTModel 实例中的相应参数。最后将加载了预训练权重的 GPTModel 实例返回。
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) 

    ax2 = ax1.twiny()  
    ax2.plot(tokens_seen, train_losses, alpha=0)  
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout() 
    plt.savefig("loss-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)