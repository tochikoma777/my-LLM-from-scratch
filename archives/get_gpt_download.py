import urllib.request
from gpt_download import download_and_load_gpt2
""" url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename) """


# settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")



""" from transformers import GPT2Model
model_names = {
    "gpt2-small (124M)": "openai-community/gpt2",
    "gpt2-medium (355M)": "openai-community/gpt2-medium",
    "gpt2-large (774M)": "openai-community/gpt2-large",
    "gpt2-xl (1558M)": "openai-community/gpt2-xl"
}

CHOOSE_MODEL = "gpt2-small (124M)"

gpt_hf = GPT2Model.from_pretrained(model_names[CHOOSE_MODEL], cache_dir="checkpoints")
gpt_hf.eval() """


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
    trust_remote_code=True  # 额外添加，避免镜像源的小兼容问题
)