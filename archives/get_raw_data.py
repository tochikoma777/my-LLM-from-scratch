# Download the raw data file from GitHub
# 可优化点：
# 1. 添加错误处理机制，使用 try-except 块来捕获可能发生的网络错误或文件写入错误，并提供相应的错误提示。
# 2. 添加文件存在检查，在下载之前检查本地是否已经存在同名文件，以避免重复下载和覆盖现有文件。
# 3. 添加下载进度显示，使用 tqdm 库或其他方法来显示下载进度，以提高用户体验。

# 1. 导入 urllib 库中的 request 模块,专门用于处理 URL 相关的请求（比如下载文件、发送 HTTP 请求等），无需额外安装，是 Python 内置的
import urllib.request
# 2. 拼接远程文件的完整 URL（分行写是为了代码可读性）
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
# 3. 定义本地保存文件的路径/名称（这里直接放在当前目录下，命名为 "the-verdict.txt"）
file_path = "the-verdict.txt"
# 4. 执行下载操作：将远程 URL 的文件保存到本地指定路径，语法：urllib.request.urlretrieve(url, filename=None, reporthook=None, data=None)
urllib.request.urlretrieve(url, file_path)
# 5. 下载完成后，打印提示信息
print(f"File downloaded successfully and saved as '{file_path}'")
