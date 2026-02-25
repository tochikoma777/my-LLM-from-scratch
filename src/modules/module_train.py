"""
模型训练模块 (Model Training Module)

本模块包含了训练 GPT 模型的完整流程，包括：
1. 数据准备与加载
2. 模型初始化与配置
3. 训练循环（前向传播、损失计算、反向传播、参数更新）
4. 模型评估与验证
5. 文本生成示例
6. 训练可视化与模型保存

支持的功能：
- 自动下载示例数据集
- 训练/验证损失监控
- 定期生成文本样本观察训练进展
- 模型检查点保存与加载
"""

import matplotlib.pyplot as plt
import os
import torch
import urllib.request
import tiktoken

# 相对导入（当作为包使用时）
from language_module import GPTModel
from data_preprocess import create_dataloader_v1
from generate_text_simple import generate_text_simple


def text_to_token_ids(text, tokenizer):
    """
    将文本字符串转换为模型输入格式的 token IDs 张量
    
    这是文本生成的预处理步骤，将自然语言转换为模型可处理的整数序列。
    
    参数:
        text (str): 输入文本字符串
        tokenizer: 分词器对象（如 tiktoken 的 Encoding）
    
    返回:
        torch.Tensor: 形状为 (1, seq_len) 的 LongTensor，批次维度为1
    
    示例:
        >>> tokenizer = tiktoken.get_encoding("gpt2")
        >>> text = "Hello world"
        >>> token_ids = text_to_token_ids(text, tokenizer)
        >>> print(token_ids.shape)  # torch.Size([1, 2])
        >>> print(token_ids)        # tensor([[15496, 995]])
    """
    # 使用分词器编码文本，得到整数列表
    encoded = tokenizer.encode(text)
    
    # 转换为 PyTorch 张量并添加批次维度（unsqueeze 在第0维）
    # 形状从 (seq_len,) 变为 (1, seq_len)，符合模型输入要求 (batch, seq)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    将模型输出的 token IDs 张量转换回可读文本字符串
    
    这是文本生成的后处理步骤，将模型的整数输出还原为自然语言。
    
    参数:
        token_ids (torch.Tensor): 模型输出的 token IDs，形状通常为 (1, seq_len) 或 (batch, seq)
        tokenizer: 分词器对象
    
    返回:
        str: 解码后的文本字符串
    
    示例:
        >>> token_ids = torch.tensor([[15496, 995, 0]])  # "Hello world<|endoftext|>"
        >>> text = token_ids_to_text(token_ids, tokenizer)
        >>> print(text)  # "Hello world<|endoftext|>"
    """
    # 移除批次维度，将张量展平为一维列表
    # squeeze(0) 移除第0维（批次维度），如果该维度大小为1
    flat = token_ids.squeeze(0)
    
    # 转换为 Python 列表并解码为文本
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    计算单个批次的交叉熵损失
    
    这是训练过程中的核心函数，计算模型预测与真实标签之间的差异。
    
    参数:
        input_batch (torch.Tensor): 输入序列，形状 (batch_size, seq_len)
        target_batch (torch.Tensor): 目标序列，形状 (batch_size, seq_len)
        model (nn.Module): GPT 模型实例
        device (torch.device): 计算设备（CPU 或 CUDA）
    
    返回:
        torch.Tensor: 标量损失值（平均交叉熵损失）
    
    计算细节:
        1. 将输入和目标移动到指定设备
        2. 模型前向传播得到 logits，形状 (batch, seq_len, vocab_size)
        3. 使用 cross_entropy 计算损失，需要展平为二维张量:
           - logits: (batch*seq_len, vocab_size)
           - targets: (batch*seq_len,)
    """
    # 将数据移动到计算设备（GPU/CPU）
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    # 模型前向传播，获取预测 logits
    # logits 形状: (batch_size, seq_len, vocab_size)
    logits = model(input_batch)
    
    # 计算交叉熵损失
    # flatten(0, 1) 将前两个维度合并: (batch, seq, vocab) -> (batch*seq, vocab)
    # target_batch.flatten() 将目标展平: (batch, seq) -> (batch*seq,)
    # cross_entropy 默认使用 mean 归约，返回标量损失
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),      # 展平的预测分数，形状 (batch*seq, vocab)
        target_batch.flatten()      # 展平的真实标签，形状 (batch*seq,)
    )
    
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    计算整个数据加载器的平均损失
    
    用于评估模型在训练集或验证集上的整体表现。
    
    参数:
        data_loader (DataLoader): PyTorch 数据加载器
        model (nn.Module): GPT 模型实例
        device (torch.device): 计算设备
        num_batches (int, optional): 限制计算的批次数量，用于快速评估。
                                    若为 None，则计算所有批次。
    
    返回:
        float: 平均损失值
    
    注意事项:
        - 使用 model.eval() 模式（由调用者控制）禁用 Dropout
        - 使用 torch.no_grad()（由调用者控制）禁用梯度计算，节省内存
    """
    # 初始化总损失
    total_loss = 0.0
    
    # 处理空数据加载器的情况
    if len(data_loader) == 0:
        return float("nan")  # 返回 NaN 表示无效值
    
    # 确定要计算的批次数量
    if num_batches is None:
        num_batches = len(data_loader)  # 计算全部
    else:
        # 限制为实际可用的批次数量，避免索引越界
        num_batches = min(num_batches, len(data_loader))
    
    # 遍历数据加载器
    for i, (input_batch, target_batch) in enumerate(data_loader):
        # 只计算指定的前 num_batches 个批次
        if i < num_batches:
            # 计算当前批次的损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 累加损失（使用 .item() 获取 Python 浮点数）
            total_loss += loss.item()
        else:
            # 达到指定批次后提前退出
            break
    
    # 返回平均损失
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    评估模型在训练集和验证集上的性能
    
    在训练过程中定期调用，监控过拟合情况。
    
    参数:
        model (nn.Module): GPT 模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        device (torch.device): 计算设备
        eval_iter (int): 评估时使用的批次数量（用于快速评估）
    
    返回:
        tuple: (train_loss, val_loss) 训练损失和验证损失
    
    流程:
        1. 切换到评估模式 (eval)，禁用 Dropout
        2. 禁用梯度计算 (no_grad)，节省内存加速计算
        3. 分别计算训练集和验证集损失
        4. 切换回训练模式 (train)
    """
    # 设置模型为评估模式
    # 这会禁用 Dropout 和 BatchNorm 的训练行为
    model.eval()
    
    # 禁用梯度计算上下文管理器
    # 减少内存使用并加速计算，因为不需要反向传播
    with torch.no_grad():
        # 计算训练集损失（使用部分批次加速）
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        # 计算验证集损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    
    # 切换回训练模式，重新启用 Dropout
    model.train()
    
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    生成文本样本并打印，用于观察训练进展
    
    在训练过程中定期调用，直观地查看模型学习效果。
    
    参数:
        model (nn.Module): GPT 模型
        tokenizer: 分词器
        device (torch.device): 计算设备
        start_context (str): 生成提示词（prompt），模型从此开始续写
    
    示例输出:
        Input: "Every effort moves you"
        Output: "Every effort moves you forward in the journey of life..."
    """
    # 切换到评估模式
    model.eval()
    
    # 获取模型的最大上下文长度（从位置编码权重形状推断）
    # pos_emb.weight 形状: (context_length, emb_dim)
    context_size = model.pos_emb.weight.shape[0]
    
    # 将起始文本编码为 token IDs 并移动到设备
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    
    # 禁用梯度计算进行生成
    with torch.no_grad():
        # 使用简单贪婪解码生成文本
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,      # 生成 50 个新 token
            context_size=context_size
        )
        
        # 将生成的 token IDs 解码为文本
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        
        # 打印生成的文本，将换行符替换为空格便于阅读
        print(decoded_text.replace("\n", " "))
    
    # 切换回训练模式
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """
    简化版 GPT 模型训练函数
    
    实现标准的训练循环，包含定期评估和文本生成示例。
    
    参数:
        model (nn.Module): 待训练的 GPT 模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        optimizer (Optimizer): PyTorch 优化器（如 AdamW）
        device (torch.device): 训练设备
        num_epochs (int): 训练轮数（完整遍历数据集的次数）
        eval_freq (int): 评估频率（每多少个全局步骤评估一次）
        eval_iter (int): 每次评估使用的批次数量
        start_context (str): 文本生成的提示词
        tokenizer: 分词器
    
    返回:
        tuple: (train_losses, val_losses, track_tokens_seen)
               训练损失列表、验证损失列表、已见 token 数列表
    
    训练流程:
        For each epoch:
            For each batch:
                1. 梯度清零
                2. 前向传播计算损失
                3. 反向传播计算梯度
                4. 优化器更新参数
                5. 定期评估并记录损失
            每个 epoch 结束生成文本样本
    """
    # 初始化跟踪列表
    train_losses = []      # 记录训练损失
    val_losses = []        # 记录验证损失
    track_tokens_seen = [] # 记录已处理的 token 数量
    
    # 初始化计数器
    tokens_seen = 0        # 累计处理的 token 总数
    global_step = -1       # 全局步骤计数器（从 -1 开始，第一次评估时为 0）
    
    # ===== 外层循环：遍历多个 epoch =====
    for epoch in range(num_epochs):
        # 设置模型为训练模式（启用 Dropout 等）
        model.train()
        
        # ===== 内层循环：遍历训练数据 =====
        for input_batch, target_batch in train_loader:
            # 步骤 1: 梯度清零
            # 防止梯度累积，每次迭代前清除旧梯度
            optimizer.zero_grad()
            
            # 步骤 2: 计算损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            # 步骤 3: 反向传播
            # 计算损失相对于所有可训练参数的梯度
            loss.backward()
            
            # 步骤 4: 参数更新
            # 优化器根据梯度更新模型参数
            optimizer.step()
            
            # 更新统计信息
            # numel() 返回张量中元素的总数（batch_size * seq_len）
            tokens_seen += input_batch.numel()
            global_step += 1
            
            # 步骤 5: 定期评估
            # 每 eval_freq 个全局步骤进行一次评估
            if global_step % eval_freq == 0:
                # 评估模型，获取当前训练和验证损失
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                
                # 记录评估结果
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                # 打印训练进度
                # 格式: Ep 1 (Step 000010): Train loss 2.523, Val loss 2.651
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        # ===== 每个 epoch 结束：生成文本样本 =====
        # 观察模型当前的学习效果和生成能力
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    
    # 返回训练历史记录
    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    绘制训练过程中的损失变化曲线
    
    创建双坐标轴图表，同时展示损失随 epochs 和 tokens 的变化。
    
    参数:
        epochs_seen (list/tensor): 评估时对应的 epoch 数
        tokens_seen (list): 评估时已处理的 token 总数
        train_losses (list): 训练损失值列表
        val_losses (list): 验证损失值列表
    
    图表特征:
        - 主 x 轴: Epochs
        - 次 x 轴（顶部）: Tokens seen
        - y 轴: Loss
        - 两条曲线: 训练损失（实线）、验证损失（虚线）
    """
    # 创建图形和主坐标轴
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 在主坐标轴上绘制损失曲线
    # 训练损失：实线
    ax1.plot(epochs_seen, train_losses, label="Training loss", linewidth=2)
    # 验证损失：点划线
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss", linewidth=2)
    
    # 设置主坐标轴标签
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)  # 添加网格线
    
    # 创建共享 y 轴的次 x 轴（顶部）
    # 用于显示 tokens_seen 刻度
    ax2 = ax1.twiny()
    
    # 绘制不可见的线，仅用于对齐刻度
    # alpha=0 使线完全透明
    ax2.plot(tokens_seen, train_losses, alpha=0)
    
    # 设置次坐标轴标签
    ax2.set_xlabel("Tokens seen", fontsize=12)
    
    # 自动调整布局，防止标签重叠
    fig.tight_layout()
    
    # 显示图表
    plt.show()


def main(gpt_config, settings):
    """
    主训练流程函数
    
    整合数据准备、模型初始化、训练和保存的完整流程。
    
    参数:
        gpt_config (dict): GPT 模型架构配置
        settings (dict): 训练超参数配置
    
    返回:
        tuple: (train_losses, val_losses, tokens_seen, model)
               训练历史记录和训练好的模型
    """
    # 设置随机种子，确保实验可复现
    torch.manual_seed(123)
    
    # 自动检测并选择计算设备
    # 优先使用 CUDA GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # ===== 数据准备 =====
    # 定义数据文件路径和下载 URL
    file_path = "the-verdict.txt"
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
           "main/ch02/01_main-chapter-code/the-verdict.txt")
    
    # 检查本地是否已有数据文件
    if not os.path.exists(file_path):
        print(f"正在下载数据文件...")
        # 使用 urllib 下载文本数据
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        # 保存到本地
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
        print(f"数据已保存到: {file_path}")
    else:
        print(f"找到本地数据文件: {file_path}")
        # 从本地读取
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    
    print(f"文本总长度: {len(text_data)} 字符")
    
    # ===== 模型初始化 =====
    # 创建 GPT 模型实例
    model = GPTModel(gpt_config)
    # 将模型移动到计算设备
    model.to(device)
    
    # 初始化 AdamW 优化器
    # AdamW 是 Adam 的改进版，正确实现权重衰减（L2 正则化）
    optimizer = torch.optim.AdamW(
        model.parameters(),                    # 要优化的参数
        lr=settings["learning_rate"],          # 学习率
        weight_decay=settings["weight_decay"]  # 权重衰减系数（防止过拟合）
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # ===== 数据集划分 =====
    # 90% 训练集，10% 验证集
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    
    # 创建训练数据加载器
    # stride 设置为 context_length 确保无重叠，覆盖整个数据集
    train_loader = create_dataloader_v1(
        text_data[:split_idx],                 # 训练集文本
        batch_size=settings["batch_size"],     # 批次大小
        max_length=gpt_config["context_length"],  # 序列长度
        stride=gpt_config["context_length"],   # 步长（无重叠）
        drop_last=True,                        # 丢弃不完整批次
        shuffle=True,                          # 打乱数据
        num_workers=0                          # 数据加载进程数
    )
    
    # 创建验证数据加载器
    val_loader = create_dataloader_v1(
        text_data[split_idx:],                 # 验证集文本
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,                       # 保留所有验证样本
        shuffle=False,                         # 验证集不打乱
        num_workers=0
    )
    
    print(f"训练批次数量: {len(train_loader)}")
    print(f"验证批次数量: {len(val_loader)}")
    
    # 初始化 GPT-2 分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # ===== 开始训练 =====
    print("\n开始训练...")
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=settings["num_epochs"],     # 训练轮数
        eval_freq=5,                           # 每 5 步评估一次
        eval_iter=1,                           # 评估时使用 1 个批次
        start_context="Every effort moves you",  # 文本生成提示词
        tokenizer=tokenizer
    )
    
    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
    # ===== 模型配置 =====
    # GPT-2 Small (124M) 配置
    GPT_CONFIG_124M = {
        "vocab_size": 50257,      # 词汇表大小（GPT-2 使用 BPE，词汇表 50257）
        "context_length": 256,    # 上下文长度（这里使用 256 而非标准的 1024，节省计算资源）
        "emb_dim": 768,           # 嵌入维度（模型宽度）
        "n_heads": 12,            # 注意力头数
        "n_layers": 12,           # Transformer 块数量（模型深度）
        "drop_rate": 0.1,         # Dropout 概率
        "qkv_bias": False         # 是否使用 QKV 偏置（GPT-2 不使用）
    }
    
    # ===== 训练超参数配置 =====
    OTHER_SETTINGS = {
        "learning_rate": 5e-4,    # 学习率（AdamW 常用范围：1e-5 ~ 1e-3）
        "num_epochs": 10,         # 训练轮数
        "batch_size": 2,          # 批次大小（根据 GPU 内存调整）
        "weight_decay": 0.1       # 权重衰减（L2 正则化系数）
    }
    
    # 执行训练流程
    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)
    
    # ===== 可视化训练过程 =====
    # 生成 epoch 张量（均匀分布在训练轮数之间）
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    
    # 绘制损失曲线
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    
    # 保存图表为 PDF
    plt.savefig("loss.pdf", dpi=300, bbox_inches='tight')
    print("训练曲线已保存到 loss.pdf")
    
    # ===== 保存模型 =====
    # 保存模型状态字典（仅参数，不包含结构）
    torch.save(model.state_dict(), "model.pth")
    print("模型已保存到 model.pth")
    
    # ===== 加载模型示例 =====
    # 创建新模型实例
    model = GPTModel(GPT_CONFIG_124M)
    # 加载保存的权重
    # weights_only=True 是安全做法，防止加载恶意 pickle 数据
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    print("模型加载成功，可用于推理或继续训练")