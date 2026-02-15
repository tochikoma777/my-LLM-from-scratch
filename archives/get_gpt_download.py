# 说明：
# 该代码用于下载并加载 GPT-2 模型的权重和配置文件。
   

import urllib.request
# 1.下载 gpt_download.py 文件 
from gpt_download import download_and_load_gpt2
url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
# 2.从 URL 中提取文件名并下载文件
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)

# 3.下载并加载 GPT-2 模型的权重和配置文件
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")



""" # 说明：
# 该代码用于设置环境变量以解决在中国大陆下载 Hugging Face 模型时遇到的 SSL 验证问题，并指定使用国内镜像源以加速下载过程。
import os
import ssl
import urllib3

# 1. 关闭 SSL 验证
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# 2. 设置 Hugging Face 国内镜像源（关键）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 3. 关闭 oneDNN 提示
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 4. 原有加载模型代码
from transformers import GPT2Model

model_names = {"124M": "openai-community/gpt2"}
CHOOSE_MODEL = "124M"
# cache_dir 指定本地缓存目录，避免重复下载
gpt_hf = GPT2Model.from_pretrained(
    model_names[CHOOSE_MODEL], 
    cache_dir="checkpoints",
    trust_remote_code=True
) """