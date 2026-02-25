"""
数据预处理模块 (Data Preprocessing Module)

本模块负责将原始文本数据转换为模型可以接受的输入格式。
主要功能包括：
1. 使用 GPT-2 分词器对文本进行编码
2. 通过滑动窗口机制创建输入-目标序列对
3. 构建 PyTorch Dataset 和 DataLoader 用于批量数据加载

核心概念：
- Tokenization: 将文本转换为整数序列
- Sliding Window: 滑窗切分长文本为固定长度的训练样本
- Input-Target Pair: 语言建模的标准形式，目标是输入的下一个token
"""

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """
    GPT数据集类 (GPT Dataset Class)
    
    继承自 PyTorch 的 Dataset 类，用于存储和管理 GPT 模型的训练数据。
    通过滑动窗口机制将长文本切分为多个固定长度的训练样本。
    
    属性:
        input_ids (list): 存储输入序列的列表，每个元素是形状为 (max_length,) 的 torch.Tensor
        target_ids (list): 存储目标序列的列表，每个元素是形状为 (max_length,) 的 torch.Tensor
    
    原理说明:
        在自回归语言模型中，我们需要预测下一个token。因此：
        - 输入序列: [t0, t1, t2, ..., tn-1]
        - 目标序列: [t1, t2, t3, ..., tn]
        即目标序列是输入序列整体右移一位的结果
    """
    
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        初始化 GPT 数据集
        
        参数:
            txt (str): 原始文本字符串，用于训练的语言数据
            tokenizer: 分词器对象，用于将文本转换为token IDs
            max_length (int): 每个训练样本的序列长度（上下文窗口大小）
            stride (int): 滑动窗口的步长，控制相邻样本之间的重叠程度
        
        示例:
            若 max_length=4, stride=1:
            文本: "Hello world this is GPT"
            Token IDs: [1, 2, 3, 4, 5, 6]
            
            样本1: 输入=[1,2,3,4], 目标=[2,3,4,5]
            样本2: 输入=[2,3,4,5], 目标=[3,4,5,6]  (stride=1，重叠3个token)
            
            若 stride=4:
            样本1: 输入=[1,2,3,4], 目标=[2,3,4,5]
            样本2: 输入=[5,6,...], ...  (无重叠，但可能丢失部分信息)
        """
        super().__init__()
        
        # 初始化存储容器
        # self.input_ids 存储输入序列（模型看到的上下文）
        self.input_ids = []
        # self.target_ids 存储目标序列（模型需要预测的下一个token）
        self.target_ids = []

        # 第一步：使用分词器对整个文本进行编码
        # allowed_special 参数允许编码特殊的控制token，如<|endoftext|>表示文本结束
        # <|endoftext|> 在 GPT-2 中的 ID 通常是 50256
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        # 打印编码信息，帮助调试和理解数据规模
        print(f"[数据集初始化] 文本总长度: {len(txt)} 字符")
        print(f"[数据集初始化] Token 总数: {len(token_ids)}")
        print(f"[数据集初始化] 预计生成样本数: ~{(len(token_ids) - max_length) // stride}")

        # 第二步：使用滑动窗口切分文本，生成训练样本对
        # range(start, stop, step): 从0开始，到 len(token_ids) - max_length 结束，步长为 stride
        # 确保每个输入序列都有对应的目标序列（目标长度也是 max_length）
        for i in range(0, len(token_ids) - max_length, stride):
            # 提取输入块：从位置 i 开始，长度为 max_length 的连续token序列
            # 例如: token_ids[0:4] 获取前4个token
            input_chunk = token_ids[i:i + max_length]
            
            # 提取目标块：从位置 i+1 开始，长度为 max_length
            # 这是输入块整体向右偏移一位的结果，实现"预测下一个token"的目标
            # 例如: 输入=[1,2,3,4]，目标=[2,3,4,5]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            
            # 将 Python 列表转换为 PyTorch 张量
            # dtype 默认为 torch.int64 (LongTensor)，适合作为 Embedding 层的索引
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        print(f"[数据集初始化] 实际生成样本数: {len(self.input_ids)}")

    def __len__(self):
        """
        返回数据集中的样本总数
        
        PyTorch Dataset 必须实现的方法，用于 DataLoader 确定数据集大小
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        根据索引获取单个样本
        
        参数:
            idx (int): 样本索引，范围在 [0, len(self)-1]
        
        返回:
            tuple: (input_tensor, target_tensor) 输入和目标张量对
        
        PyTorch Dataset 必须实现的方法，用于支持索引访问和 DataLoader 批量加载
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    创建 GPT 数据加载器 (Create GPT DataLoader)
    
    工厂函数，用于快速创建配置好的 DataLoader 实例。
    自动初始化 GPT-2 分词器并构建数据集。
    
    参数:
        txt (str): 原始文本数据
        batch_size (int): 每个批次的样本数量，影响 GPU 内存使用和训练稳定性
        max_length (int): 每个序列的最大长度（上下文长度）
        stride (int): 滑动窗口步长，控制样本间重叠度
        shuffle (bool): 是否在每个 epoch 开始时打乱数据顺序，训练时通常设为 True
        drop_last (bool): 是否丢弃最后一个不完整的批次，避免批次大小不一致
        num_workers (int): 数据加载的并行进程数，0 表示在主进程加载（适合大多数情况）
    
    返回:
        DataLoader: 配置好的 PyTorch 数据加载器，可迭代生成批次数据
    
    使用示例:
        >>> with open("novel.txt", "r") as f:
        ...     text = f.read()
        >>> dataloader = create_dataloader_v1(
        ...     text, batch_size=8, max_length=512, stride=256
        ... )
        >>> for batch_idx, (inputs, targets) in enumerate(dataloader):
        ...     print(f"批次 {batch_idx}: 输入形状 {inputs.shape}")
        ...     # 训练代码...
    """
    # 初始化 GPT-2 分词器 (BPE 编码)
    # tiktoken 是 OpenAI 开发的高性能分词库，与 GPT-2/GPT-3 完全兼容
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"[DataLoader] 初始化 GPT-2 分词器，词汇表大小: {tokenizer.n_vocab}")

    # 创建数据集实例
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建 DataLoader
    # DataLoader 是 PyTorch 提供的批量数据加载工具，支持：
    # - 自动批次组合 (batching)
    # - 多进程数据加载 (num_workers)
    # - 数据打乱 (shuffle)
    # - 内存锁定 (pin_memory，加速 GPU 传输)
    dataloader = DataLoader(
        dataset=dataset,           # 数据源
        batch_size=batch_size,     # 批次大小
        shuffle=shuffle,           # 是否打乱
        drop_last=drop_last,       # 是否丢弃不完整批次
        num_workers=num_workers    # 加载进程数
    )
    
    print(f"[DataLoader] 配置完成: 批次大小={batch_size}, 序列长度={max_length}")
    
    return dataloader