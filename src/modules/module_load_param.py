"""
预训练权重加载模块 (Pretrained Weights Loader Module)

本模块实现了从 OpenAI 发布的原始 GPT-2 TensorFlow 检查点文件
加载权重到 PyTorch 模型的功能。

主要功能：
1. 自动下载 OpenAI GPT-2 预训练权重（124M/355M/774M/1558M）
2. 将 TensorFlow 检查点转换为 PyTorch 格式
3. 权重映射：将 TensorFlow 变量名对应到 PyTorch 模型参数
4. 增强版文本生成（支持 temperature 采样和 top-k 筛选）

技术细节：
- TensorFlow 检查点格式解析
- 跨框架权重转换（TF -> PyTorch）
- 命名空间映射处理
"""

import json
import numpy as np
import os
import urllib.request

import tensorflow as tf
import tiktoken
import torch
from tqdm import tqdm

from language_module import GPTModel


def text_to_token_ids(text, tokenizer):
    """
    将文本转换为 token IDs 张量（辅助函数）
    
    与 training_module.py 中的实现相同，确保一致性。
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    将 token IDs 张量转换为文本（辅助函数）
    
    与 training_module.py 中的实现相同。
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def download_and_load_gpt2(model_size, models_dir):
    """
    下载并加载 GPT-2 预训练权重
    
    从 OpenAI 官方仓库下载指定大小的 GPT-2 模型文件，
    并解析为可用的设置和参数字典。
    
    参数:
        model_size (str): 模型大小，可选 "124M", "355M", "774M", "1558M"
        models_dir (str): 本地存储目录
    
    返回:
        tuple: (settings, params)
               - settings (dict): 模型超参数配置
               - params (dict): 嵌套字典形式的模型权重
    
    文件说明:
        - checkpoint: TF 检查点信息
        - encoder.json: BPE 编码器词汇表
        - hparams.json: 超参数（层数、维度等）
        - model.ckpt.*: 模型权重文件（数据、索引、元信息）
        - vocab.bpe: BPE 合并规则
    """
    # 验证模型大小参数
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"不支持的模型大小: {model_size}。可选: {allowed_sizes}")
    
    # 构建本地模型目录路径
    model_dir = os.path.join(models_dir, model_size)
    
    # OpenAI GPT-2 权重存储在 Azure Blob 存储
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    
    # 需要下载的文件列表
    filenames = [
        "checkpoint",                    # 检查点信息
        "encoder.json",                  # BPE 编码器
        "hparams.json",                  # 超参数
        "model.ckpt.data-00000-of-00001", # 权重数据（主要文件）
        "model.ckpt.index",              # 权重索引
        "model.ckpt.meta",               # 计算图元数据
        "vocab.bpe"                      # BPE 规则
    ]
    
    # 创建目录并下载文件
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"正在下载 GPT-2 {model_size} 模型文件...")
    for filename in filenames:
        # 构建 URL 和本地路径
        file_url = f"{base_url}/{model_size}/{filename}"
        file_path = os.path.join(model_dir, filename)
        
        # 下载文件（带进度条）
        download_file(file_url, file_path)
    
    # 加载模型设置（超参数）
    hparams_path = os.path.join(model_dir, "hparams.json")
    with open(hparams_path, "r") as f:
        settings = json.load(f)
    
    # 查找最新的检查点文件
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    
    # 从 TensorFlow 检查点加载权重参数
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    
    print(f"成功加载 GPT-2 {model_size} 权重")
    return settings, params


def download_file(url, destination):
    """
    带进度条的文件下载函数
    
    支持断点续传（如果文件已存在且大小匹配则跳过）。
    
    参数:
        url (str): 文件 URL
        destination (str): 本地保存路径
    """
    # 发送 HTTP GET 请求
    with urllib.request.urlopen(url) as response:
        # 从响应头获取文件总大小（字节）
        file_size = int(response.headers.get("Content-Length", 0))
        
        # 检查本地文件是否已存在且完整
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"文件已存在且完整: {os.path.basename(destination)}")
                return  # 跳过下载
        
        # 分块读取的块大小（1KB）
        block_size = 1024
        
        # 从 URL 提取文件名用于进度条描述
        progress_bar_description = os.path.basename(url)
        
        # 使用 tqdm 创建进度条
        with tqdm(total=file_size, unit="iB", unit_scale=True, 
                  desc=progress_bar_description) as progress_bar:
            
            # 以二进制写模式打开目标文件
            with open(destination, "wb") as file:
                # 循环读取数据块直到文件结束
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break  # 下载完成
                    file.write(chunk)
                    progress_bar.update(len(chunk))  # 更新进度条


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """
    从 TensorFlow 检查点加载并转换权重参数
    
    解析 TF 检查点的变量名，构建嵌套字典结构的参数。
    
    参数:
        ckpt_path (str): TensorFlow 检查点路径前缀
        settings (dict): 模型超参数（用于确定层数等）
    
    返回:
        dict: 嵌套字典，结构如下：
        {
            "wpe": 位置编码权重,
            "wte": 词嵌入权重,
            "blocks": [
                {
                    "attn": {...},  # 注意力层权重
                    "mlp": {...},   # MLP 层权重
                    "ln_1": {...},  # 第一层归一化
                    "ln_2": {...}   # 第二层归一化
                },
                ... # 共 n_layer 个块
            ],
            "g": 最终层归一化 scale,
            "b": 最终层归一化 shift
        }
    
    TensorFlow 变量命名规则示例：
        model/h0/attn/c_attn/w:0  -> 第0层注意力 QKV 权重
        model/h1/mlp/c_fc/b:0     -> 第1层 MLP 第一层偏置
    """
    # 初始化参数字典，为每个 Transformer 块创建空字典
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    
    # 遍历检查点中的所有变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量数组并移除单例维度（如 (768, 1) -> (768,)）
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        
        # 处理变量名，移除 'model/' 前缀并按 '/' 分割
        # 例如: "model/h0/attn/c_attn/w" -> ["h0", "attn", "c_attn", "w"]
        variable_name_parts = name.split("/")[1:]
        
        # 确定目标字典（根字典或特定块的字典）
        target_dict = params
        
        # 检查是否是 Transformer 块的变量（以 'h' 开头，如 h0, h1...）
        if variable_name_parts[0].startswith("h"):
            # 提取层号，例如 "h0" -> 0
            layer_number = int(variable_name_parts[0][1:])
            # 切换到对应层的字典
            target_dict = params["blocks"][layer_number]
        
        # 遍历变量名的中间部分，创建嵌套字典结构
        # 例如: ["attn", "c_attn"] -> params["blocks"][0]["attn"]["c_attn"]
        for key in variable_name_parts[1:-1]:
            # setdefault: 如果 key 不存在则创建空字典，返回该字典
            target_dict = target_dict.setdefault(key, {})
        
        # 将变量数组存储到最后一个 key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array
    
    return params


def assign(left, right):
    """
    辅助函数：将 numpy 数组赋值给 PyTorch 参数
    
    检查形状匹配后，创建 PyTorch Parameter。
    
    参数:
        left (torch.Tensor): 目标 PyTorch 参数（用于获取形状）
        right (np.ndarray): 源 numpy 数组
    
    返回:
        torch.nn.Parameter: 可训练的 PyTorch 参数
    """
    if left.shape != right.shape:
        raise ValueError(
            f"形状不匹配。目标: {left.shape}, 源: {right.shape}"
        )
    # 将 numpy 数组转换为 PyTorch 张量，并包装为 Parameter
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """
    将加载的权重参数赋值给 PyTorch GPT 模型
    
    这是权重转换的核心函数，处理 TensorFlow 和 PyTorch 之间的命名和维度差异。
    
    参数:
        gpt (GPTModel): 目标 PyTorch 模型
        params (dict): 从 TensorFlow 检查点加载的参数字典
    
    关键转换说明：
        1. 嵌入层: 直接赋值
        2. 注意力层: 需要分割 QKV 合并权重（TF 是合并存储，PyTorch 是分开）
        3. 线性层: 需要转置（TF 使用 (in, out)，PyTorch 使用 (out, in)）
        4. 层归一化: 参数名映射（g->scale, b->shift）
    """
    # ===== 嵌入层 =====
    # 位置编码权重 (Positional Embeddings)
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    # 词嵌入权重 (Token Embeddings)
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    # ===== Transformer 块 =====
    for b in range(len(params["blocks"])):
        # --- 注意力层 QKV 权重分割 ---
        # TensorFlow 将 Q、K、V 的权重合并存储为一个大矩阵 (3*d_out, d_out)
        # 需要分割为三个独立的矩阵
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"],  # 合并的权重
            3,  # 分割为 3 份
            axis=-1  # 在最后一个维度分割
        )
        
        # 赋值查询 (Query) 权重，注意转置（维度对齐）
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        # 赋值键 (Key) 权重
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        # 赋值值 (Value) 权重
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        
        # --- 注意力层 QKV 偏置分割 ---
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"],  # 合并的偏置
            3, axis=-1)
        
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)
        
        # --- 注意力输出投影层 ---
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)  # 转置
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
        
        # --- 前馈网络 (FFN) 第一层 ---
        # c_fc: 全连接层（扩展维度）
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        
        # --- 前馈网络 (FFN) 第二层 ---
        # c_proj: 投影层（收缩维度）
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        # --- 层归一化参数 ---
        # ln_1: 注意力前的归一化
        # TensorFlow 使用 "g" (gain) 和 "b" (bias)，对应 PyTorch 的 "scale" 和 "shift"
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        
        # ln_2: 前馈网络前的归一化
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])
    
    # ===== 最终层归一化 =====
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    
    # ===== 输出层（与词嵌入共享权重）=====
    # GPT-2 使用权重绑定（Weight Tying），输出层与输入嵌入共享参数
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
    
    print("权重加载完成，所有参数已成功映射到 PyTorch 模型")


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    增强版文本生成函数（支持 Temperature 采样和 Top-k 筛选）
    
    相比 generate_text_simple 的贪婪解码，此函数实现了更灵活的采样策略，
    可以生成更多样化、更自然的文本。
    
    参数:
        model (nn.Module): GPT 模型
        idx (torch.Tensor): 初始 token 序列，形状 (batch_size, seq_len)
        max_new_tokens (int): 要生成的新 token 数量
        context_size (int): 模型最大上下文长度
        temperature (float): 温度参数，控制随机性
                             - 0.0: 贪婪解码（确定性）
                             - <1.0: 保守采样，更确定
                             - >1.0: 激进采样，更多样
        top_k (int, optional): Top-k 筛选，只从概率最高的 k 个 token 中采样
                               None 表示不筛选（使用全部词汇表）
        eos_id (int, optional): 结束符 token ID，遇到则停止生成
    
    返回:
        torch.Tensor: 生成的完整序列
    
    采样策略说明:
        1. Top-k: 限制候选池，避免选择概率极低的 token（减少无意义输出）
        2. Temperature: 调整概率分布的"尖锐程度"
           - 低温: 分布更尖锐，高概率 token 更容易被选中
           - 高温: 分布更平缓，增加随机性
    """
    # 自回归生成循环
    for _ in range(max_new_tokens):
        # 裁剪上下文，只保留最后 context_size 个 token
        idx_cond = idx[:, -context_size:]
        
        # 获取模型预测（禁用梯度）
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 只关注最后一个时间步的预测
        logits = logits[:, -1, :]  # 形状: (batch_size, vocab_size)
        
        # ===== Top-k 筛选 =====
        if top_k is not None:
            # 获取概率最高的 top_k 个值及其索引
            top_logits, _ = torch.topk(logits, top_k)
            # 获取第 k 高的值作为阈值
            min_val = top_logits[:, -1]
            # 将所有小于阈值的 logits 设为 -inf（概率变为0）
            logits = torch.where(
                logits < min_val, 
                torch.tensor(float('-inf')).to(logits.device), 
                logits
            )
        
        # ===== Temperature 缩放 =====
        if temperature > 0.0:
            # 温度缩放: logits /= temperature
            # 温度 > 1: 分布更平缓（增加随机性）
            # 温度 < 1: 分布更尖锐（减少随机性）
            logits = logits / temperature
            
            # 计算概率分布
            probs = torch.softmax(logits, dim=-1)
            
            # 从多项分布中采样（根据概率随机选择）
            idx_next = torch.multinomial(probs, num_samples=1)
        
        else:
            # Temperature = 0: 贪婪解码，选择概率最高的 token
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        # 检查是否生成结束符
        if eos_id is not None and idx_next.item() == eos_id:
            break  # 提前结束生成
        
        # 将新 token 追加到序列
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx


def main(gpt_config, input_prompt, model_size):
    """
    主函数：加载预训练 GPT-2 并生成文本
    
    参数:
        gpt_config (dict): 模型架构配置
        input_prompt (str): 输入提示词
        model_size (str): 模型大小（"124M", "355M", "774M", "1558M"）
    """
    # 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 下载并加载预训练权重
    print(f"正在加载 GPT-2 {model_size} 权重...")
    settings, params = download_and_load_gpt2(
        model_size=model_size, 
        models_dir="gpt2"
    )
    
    # 创建模型实例
    gpt = GPTModel(gpt_config)
    
    # 加载权重到模型
    load_weights_into_gpt(gpt, params)
    
    # 移动到设备并设置为评估模式
    gpt.to(device)
    gpt.eval()
    
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 设置随机种子（可复现性）
    torch.manual_seed(123)
    
    # 编码输入提示词
    input_tensor = text_to_token_ids(input_prompt, tokenizer).to(device)
    
    print(f"\n输入提示: '{input_prompt}'")
    print("生成文本...")
    
    # 生成文本
    token_ids = generate(
        model=gpt,
        idx=input_tensor,
        max_new_tokens=25,           # 生成 25 个新 token
        context_size=gpt_config["context_length"],
        top_k=50,                    # Top-50 筛选
        temperature=1.0              # 标准温度（平衡随机性和质量）
    )
    
    # 解码并打印结果
    output_text = token_ids_to_text(token_ids, tokenizer)
    print(f"\n输出文本:\n{output_text}")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(123)
    
    # 配置
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves you"
    
    # 基础配置（所有 GPT-2 模型共享）
    BASE_CONFIG = {
        "vocab_size": 50257,      # 词汇表大小
        "context_length": 1024,   # 上下文长度（标准 GPT-2 使用 1024）
        "drop_rate": 0.0,         # 推理时 Dropout 设为 0
        "qkv_bias": True          # 预训练模型使用偏置
    }
    
    # 不同大小模型的特定配置
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    
    # 提取模型大小标识（如 "124M"）
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    
    # 合并配置
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    
    # 运行主函数
    main(BASE_CONFIG, INPUT_PROMPT, model_size)