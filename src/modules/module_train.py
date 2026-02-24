# 这个模块包含了训练GPT模型的核心函数，包括损失计算、模型评估、文本生成和训练循环等功能。
# 这些函数负责处理训练数据、计算损失、评估模型性能，并在训练过程中生成示例文本以观察模型的学习进展。


import matplotlib.pyplot as plt
import os
import torch
import urllib.request
import tiktoken

from language_module import GPTModel
from data_preprocess import create_dataloader_v1
from generate_text_simple import generate_text_simple


# 将文本转换为token ids的函数，使用GPT-2的分词器进行编码，并将结果转换为PyTorch张量
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


# 将token ids转换回文本的函数，使用GPT-2的分词器进行解码，将生成的token ids转换为可读文本
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


# 计算一个批次的损失，输入是一个输入批次和一个目标批次，以及模型和设备信息，输出是该批次的交叉熵损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


# 计算一个数据加载器的平均损失，输入是一个数据加载器、模型、设备信息和可选的批次数量，输出是该数据加载器的平均损失
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


# 评估模型性能的函数，输入是模型、训练和验证数据加载器、设备信息以及评估批次数量，输出是训练和验证数据加载器的平均损失
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# 生成文本并打印的函数，输入是模型、分词器、设备信息和初始上下文字符串，输出是生成的文本字符串
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
        print(decoded_text.replace("\n", " ")) 
    model.train()


# 训练模型的函数，输入是模型、训练和验证数据加载器、优化器、设备信息、训练轮数、评估频率、评估批次数量、初始上下文字符串和分词器
# 输出是训练和验证损失列表以及看到的令牌数量列表
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train() 

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() 
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() 
            optimizer.step()  
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


# 绘制训练和验证损失的函数，输入是看到的训练轮数列表、看到的令牌数量列表、训练损失列表和验证损失列表
# 输出是一个包含两条曲线（训练损失和验证损失）的图表
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny() 
    ax2.plot(tokens_seen, train_losses, alpha=0) 
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout() 
    plt.show()


def main(gpt_config, settings):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 下载文本数据，如果本地不存在则从URL下载，否则从本地文件读取
    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    # 初始化模型和优化器，创建GPT模型实例并将其移动到设备上，使用AdamW优化器优化模型参数
    model = GPTModel(gpt_config)
    model.to(device) 
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    # 创建训练和验证数据加载器，将文本数据分割为训练和验证集，并使用create_dataloader_v1函数创建数据加载器
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # 获取GPT-2的分词器实例，用于将文本转换为token ids和将token ids转换回文本
    tokenizer = tiktoken.get_encoding("gpt2")
    # 训练模型，调用train_model_simple函数进行训练，并返回训练和验证损失列表以及看到的令牌数量列表
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    
        "context_length": 256,  
        "emb_dim": 768,         
        "n_heads": 12,          
        "n_layers": 12,         
        "drop_rate": 0.1,       
        "qkv_bias": False       
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    # 训练模型并获取训练和验证损失列表以及看到的令牌数量列表
    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)

    # 绘制训练和验证损失的图表，使用plot_losses函数将训练和验证损失随看到的训练轮数和令牌数量的变化绘制成图表，并保存为PDF文件
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # 保存模型权重到本地文件，并加载模型权重到新的模型实例中，使用torch.save函数将模型的状态字典保存到"model.pth"文件中
    # 然后创建一个新的GPTModel实例并使用torch.load函数加载保存的状态字典
    torch.save(model.state_dict(), "model.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", weights_only=True))