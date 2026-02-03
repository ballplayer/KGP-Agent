import google.generativeai as genai
import os

# 1. 配置 API Key（从环境变量读取）
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️ test.py: 环境变量 GOOGLE_API_KEY 未设置。请通过 PowerShell: $env:GOOGLE_API_KEY=\"<your_key>\"")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# 2. 设置网络代理 (如果在国内，必须配置，否则会报错)
# 请将端口号 '7890' 修改为你实际使用的代理端口
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


def list_available_models():
    print(f"{'模型名称 (Model Name)':<40} | {'支持的方法 (Methods)'}")
    print("-" * 80)

    try:
        # 获取所有模型列表
        for m in genai.list_models():
            # 我们主要关心能生成内容的模型 (generateContent)
            if 'generateContent' in m.supported_generation_methods:
                methods = ", ".join(m.supported_generation_methods)
                print(f"{m.name:<40} | {methods}")

    except Exception as e:
        print(f"❌ 获取模型列表失败: {e}")
        print("请检查 API Key 是否正确，以及网络代理是否配置成功。")


if __name__ == "__main__":
    list_available_models()