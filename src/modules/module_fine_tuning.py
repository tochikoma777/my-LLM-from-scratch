##################################################################################################
# 如果在Kaggle Notebook运行，下面这块代码解决以下问题：
# Kaggle Notebook 基于 Jupyter 内核运行，启动时会自动注入 -f /path/to/kernel.json 参数以建立 IPC 连接，
# 而用户代码中的 argparse 实例未声明对该参数的处理，导致 SystemExit: 2 异常抛出。

# import sys
# if '-f' in sys.argv:
#     idx = sys.argv.index('-f')
#     sys.argv = sys.argv[:idx]  
##################################################################################################


"""
指令微调模块 (Instruction Fine-tuning Module)

本模块实现了对预训练 GPT 模型进行指令微调（Instruction Fine-tuning）的完整流程。
指令微调是将基础语言模型适配到特定任务（如问答、翻译、摘要）的关键技术。

核心功能：
1. 加载和格式化指令数据集（Alpaca 格式）
2. 自定义数据整理（Collate Function）处理变长序列
3. 屏蔽填充 token 的损失计算（避免学习无意义填充）
4. 加载预训练权重并进行微调
5. 生成回复并保存结果

数据格式（Alpaca 风格）：
    {
        "instruction": "任务描述",
        "input": "可选的上下文/问题",
        "output": "期望的回答"
    }
"""

from functools import partial
from importlib.metadata import version
import json
import os
import re
import time
import urllib.request

import matplotlib.pyplot as plt
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from language_module import GPTModel
from module_load_param import (
    download_and_load_gpt2, 
    load_weights_into_gpt, 
    text_to_token_ids, 
    token_ids_to_text,
    generate
)
from module_train import calc_loss_loader, train_model_simple


class InstructionDataset(Dataset):
    """
    指令数据集类 (Instruction Dataset)
    
    专门用于处理指令微调数据的数据集类。
    在初始化时预先将所有文本编码为 token IDs，加速训练时的数据加载。
    
    数据格式示例:
        {
            "instruction": "Translate the following English text to French",
            "input": "Hello, how are you?",
            "output": "Bonjour, comment allez-vous?"
        }
    
    格式化模板:
        Below is an instruction that describes a task. Write a response...
        
        ### Instruction:
        {instruction}
        
        ### Input:
        {input} (可选)
        
        ### Response:
        {output}
    """
    
    def __init__(self, data, tokenizer):
        """
        初始化指令数据集
        
        参数:
            data (list): 指令数据列表，每个元素是包含 instruction/input/output 的字典
            tokenizer: 分词器对象
        """
        super().__init__()
        self.data = data
        
        # 预分词：在初始化时将所有文本转换为 token IDs
        # 这样可以避免在训练循环中重复进行分词，显著加速数据加载
        self.encoded_texts = []
        
        for entry in data:
            # 格式化输入部分（指令 + 可选输入）
            instruction_plus_input = format_input(entry)
            
            # 格式化回复部分
            response_text = f"\n\n### Response:\n{entry['output']}"
            
            # 组合完整文本（输入 + 回复）
            full_text = instruction_plus_input + response_text
            
            # 编码为 token IDs 并存储
            encoded = tokenizer.encode(full_text)
            self.encoded_texts.append(encoded)
        
        # 统计信息
        lengths = [len(seq) for seq in self.encoded_texts]
        print(f"数据集统计:")
        print(f"  样本数量: {len(self.encoded_texts)}")
        print(f"  平均长度: {sum(lengths)/len(lengths):.1f} tokens")
        print(f"  最大长度: {max(lengths)} tokens")
        print(f"  最小长度: {min(lengths)} tokens")

    def __getitem__(self, index):
        """
        获取指定索引的编码后文本
        
        返回的是 token IDs 列表（非张量，由 collate_fn 处理批量化）
        """
        return self.encoded_texts[index]

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.encoded_texts)


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, 
                      allowed_max_length=None, device="cpu"):
    """
    自定义数据整理函数 (Custom Collate Function)
    
    处理变长序列的批量化，包括填充（Padding）和损失屏蔽。
    这是指令微调的关键：确保模型只在回复部分计算损失，忽略输入和填充。
    
    参数:
        batch (list): 批次数据，每个元素是一个 token ID 列表（变长）
        pad_token_id (int): 填充 token 的 ID（默认 <|endoftext|> = 50256）
        ignore_index (int): 损失函数中忽略的 label 值（PyTorch 标准做法）
        allowed_max_length (int): 最大允许序列长度，超长则截断
        device (str): 目标设备（cpu 或 cuda）
    
    返回:
        tuple: (inputs_tensor, targets_tensor)
               - inputs: 输入序列（填充后）
               - targets: 目标序列（输入部分和填充设为 ignore_index）
    
    处理逻辑:
        1. 找到批次中最长序列长度
        2. 为每个序列添加 <|endoftext|> 结束符
        3. 用 pad_token_id 填充所有序列至相同长度
        4. 创建 targets：输入右移一位（预测下一个 token）
        5. 将输入部分和填充位置的 target 设为 ignore_index（不计算损失）
    """
    # 找到批次中的最大长度（+1 是为了添加结束符）
    batch_max_length = max(len(item) + 1 for item in batch)
    
    # 准备输入和目标列表
    inputs_lst, targets_lst = [], []
    
    for item in batch:
        # 复制序列（避免修改原始数据）
        new_item = item.copy()
        
        # 添加结束符 <|endoftext|>
        new_item += [pad_token_id]
        
        # 填充序列至批次最大长度
        # 使用列表乘法创建填充部分：[pad] * (batch_max_length - len)
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        
        # 创建输入：移除最后一个 token（因为不需要预测它之后的 token）
        # 例如: [t0, t1, t2, pad] -> 输入: [t0, t1, t2]
        inputs = torch.tensor(padded[:-1])
        
        # 创建目标：右移一位（输入的第 i 个 token 预测目标的第 i 个 token）
        # 例如: 输入 [t0, t1, t2] -> 目标 [t1, t2, pad]
        targets = torch.tensor(padded[1:])
        
        # ===== 关键：屏蔽输入部分和填充部分的损失 =====
        # 创建一个掩码，标记所有填充位置
        mask = targets == pad_token_id
        
        # 找到所有填充位置的索引
        indices = torch.nonzero(mask).squeeze()
        
        # 如果有多个填充位置，将第一个之后的所有填充位置设为 ignore_index
        # 保留第一个填充位置，让模型学习何时停止生成（预测结束符）
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        
        # 可选：截断至最大允许长度（防止超长序列）
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    # 将列表堆叠为批次张量，并移动到指定设备
    # stack: 将列表中的 1D 张量堆叠为 2D 张量 (batch_size, seq_len)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    
    return inputs_tensor, targets_tensor


def download_and_load_file(file_path, url):
    """
    下载并加载 JSON 数据文件
    
    如果本地不存在则从 URL 下载，否则直接读取本地文件。
    
    参数:
        file_path (str): 本地文件路径
        url (str): 远程文件 URL
    
    返回:
        list: 解析后的 JSON 数据（通常是字典列表）
    """
    if not os.path.exists(file_path):
        print(f"正在下载数据文件: {file_path}")
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
        print(f"下载完成，已保存到: {file_path}")
    else:
        print(f"使用本地数据文件: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    # 解析 JSON 数据
    with open(file_path, "r") as file:
        data = json.load(file)
    
    return data


def format_input(entry):
    """
    格式化指令数据为统一模板
    
    将 instruction/input/output 格式转换为标准文本格式。
    
    参数:
        entry (dict): 包含 instruction, input, output 的字典
    
    返回:
        str: 格式化后的文本
    
    模板:
        Below is an instruction that describes a task...
        
        ### Instruction:
        {instruction}
        
        ### Input:
        {input} (如果 input 非空)
    """
    # 指令部分（固定前缀 + 指令内容）
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    
    # 输入部分（可选，如果 entry["input"] 非空则添加）
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    
    return instruction_text + input_text


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    绘制训练损失曲线（与 training_module.py 类似，但保存为 PDF）
    
    参数:
        epochs_seen: 训练轮数列表
        tokens_seen: 已见 token 数列表
        train_losses: 训练损失列表
        val_losses: 验证损失列表
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制损失曲线
    ax1.plot(epochs_seen, train_losses, label="Training loss", linewidth=2)
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss", linewidth=2)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 添加次坐标轴显示 tokens_seen
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 不可见，仅用于对齐刻度
    ax2.set_xlabel("Tokens seen", fontsize=12)

    fig.tight_layout()
    
    # 保存图表
    plot_name = "loss-plot-standalone.pdf"
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {plot_name}")


def main(test_mode=False):
    """
    指令微调主函数
    
    完整的微调流程，包括数据准备、模型加载、微调训练和结果保存。
    
    参数:
        test_mode (bool): 测试模式（使用极小模型和数据集，用于调试）
    """
    # ===== 打印依赖包版本（便于调试和环境复现）=====
    print("\n" + "="*50)
    print("环境信息:")
    pkgs = ["matplotlib", "tiktoken", "torch", "tqdm", "tensorflow"]
    for p in pkgs:
        try:
            print(f"  {p}: {version(p)}")
        except:
            print(f"  {p}: 未安装")
    print("="*50 + "\n")

    # ===== 数据准备 =====
    file_path = "instruction-data.json"
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
           "main/ch07/01_main-chapter-code/instruction-data.json")
    
    # 下载并加载数据
    data = download_and_load_file(file_path, url)
    
    # 数据集划分：85% 训练，10% 测试，5% 验证
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    
    # 测试模式：使用极小数据集快速验证代码
    if test_mode:
        print("\n[测试模式] 使用极小数据集")
        train_data = train_data[:10]
        val_data = val_data[:10]
        test_data = test_data[:10]
    
    print(f"数据集划分:")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(val_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    print("-"*50)

    # 初始化分词器和设备
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("-"*50)

    # 创建部分应用的 collate 函数（固定 device 和 max_length）
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )

    # 数据加载配置
    num_workers = 0  # 在 Jupyter/Windows 中建议设为 0 避免多进程问题
    batch_size = 8
    
    # 设置随机种子
    torch.manual_seed(123)

    # 创建训练数据加载器
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,       # 训练时打乱数据
        drop_last=True,     # 丢弃不完整批次
        num_workers=num_workers
    )

    # 创建验证数据加载器
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,      # 验证时不打乱
        drop_last=False,
        num_workers=num_workers
    )

    # ===== 模型加载 =====
    if test_mode:
        # 测试模式：使用极小模型（快速验证）
        print("\n[测试模式] 使用小型测试模型")
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 120,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2
        }
        model = GPTModel(BASE_CONFIG)
        model.eval()
        device = "cpu"
        CHOOSE_MODEL = "Small test model"
    
    else:
        # 生产模式：加载 OpenAI GPT-2 预训练权重
        print("\n正在加载预训练模型...")
        
        # 基础配置
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 1024,
            "drop_rate": 0.0,      # 微调时通常降低 Dropout
            "qkv_bias": True
        }
        
        # 支持的模型配置
        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }
        
        # 选择模型大小（这里使用 medium 作为平衡性能和资源的选项）
        CHOOSE_MODEL = "gpt2-medium (355M)"
        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
        
        # 提取模型大小标识并下载权重
        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(
            model_size=model_size, 
            models_dir="gpt2"
        )
        
        # 创建模型并加载权重
        model = GPTModel(BASE_CONFIG)
        load_weights_into_gpt(model, params)
        model.eval()
        model.to(device)

    print(f"已加载模型: {CHOOSE_MODEL}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("-"*50)

    # ===== 微调前评估 =====
    print("\n初始损失（微调前）:")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    
    print(f"  训练损失: {train_loss:.4f}")
    print(f"  验证损失: {val_loss:.4f}")

    # ===== 微调训练 =====
    start_time = time.time()
    
    # 优化器：AdamW，使用较小的学习率进行微调
    # 权重衰减防止过拟合
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.00005,           # 微调学习率通常比预训练小 10-100 倍
        weight_decay=0.1
    )
    
    num_epochs = 2  # 微调通常只需要少量轮次
    
    print(f"\n开始微调（{num_epochs} 轮）...")
    torch.manual_seed(123)
    
    # 训练循环
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=5,          # 每 5 步评估一次
        eval_iter=5,          # 评估时使用 5 个批次
        start_context=format_input(val_data[0]),  # 使用第一个验证样本作为生成示例
        tokenizer=tokenizer
    )
    
    # 计算训练时间
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\n微调完成，耗时: {execution_time_minutes:.2f} 分钟")

    # 绘制损失曲线
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    print("-"*50)

    # ===== 生成回复并保存 =====
    print("\n为测试集生成回复...")
    
    # 遍历测试集生成回复
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        # 格式化输入（指令 + 输入）
        input_text = format_input(entry)
        
        # 生成回复
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,                    # 最多生成 256 个 token
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256                           # 遇到结束符停止
        )
        
        # 解码生成的文本
        generated_text = token_ids_to_text(token_ids, tokenizer)
        
        # 提取回复部分（移除输入提示）
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
        
        # 保存到数据条目
        test_data[i]["model_response"] = response_text

    # 保存带回复的测试集
    test_data_path = "instruction-data-with-response-standalone.json"
    with open(test_data_path, "w", encoding="utf-8") as file:
        json.dump(test_data, file, indent=4, ensure_ascii=False)
    print(f"回复已保存到: {test_data_path}")

    # 保存微调后的模型权重
    # 文件名格式: 模型名-sft-standalone.pth (SFT = Supervised Fine-Tuning)
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft-standalone.pth"
    torch.save(model.state_dict(), file_name)
    print(f"模型已保存为: {file_name}")


if __name__ == "__main__":
    # 命令行参数解析
    import argparse
    
    parser = argparse.ArgumentParser(
        description="对 GPT 模型进行指令微调 (Instruction Fine-tuning)"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("启用测试模式：使用极小的模型和数据集（10条），"
              "用于快速验证代码正确性，不进行实际训练。")
    )
    args = parser.parse_args()

    # 运行主函数
    main(args.test_mode)