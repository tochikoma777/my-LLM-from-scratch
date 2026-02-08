# Download the raw data file from GitHub

# 1. 导入 urllib 库中的 request 模块,专门用于处理 URL 相关的请求（比如下载文件、发送 HTTP 请求等），无需额外安装，是 Python 内置的
import urllib.request
# 2. 拼接远程文件的完整 URL（分行写是为了代码可读性）
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
# 3. 定义本地保存文件的路径/名称（这里直接放在当前目录下，命名为 "the-verdict.txt"）
file_path = "the-verdict.txt"
# 4. 执行下载操作：将远程 URL 的文件保存到本地指定路径
urllib.request.urlretrieve(url, file_path)
# 5. 下载完成后，打印提示信息
print(f"File downloaded successfully and saved as '{file_path}'")