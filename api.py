import os
import google.generativeai as genai
import PIL.Image

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
# 1. 配置 API Key（从环境变量读取，避免硬编码）
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️ api.py: 环境变量 GOOGLE_API_KEY 未设置。调用 LLM/Embedding 会失败，请通过 PowerShell 设置：$env:GOOGLE_API_KEY=\"<your_key>\"")
else:
    genai.configure(api_key=GOOGLE_API_KEY)


def summarize_image_and_text(image_path, user_text):
    try:
        # 2. 选择模型
        # 'gemini-1.5-flash' 是目前性价比最高的多模态模型，速度快且支持图片
        # 如果任务非常复杂，也可以使用 'gemini-1.5-pro'
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("model:", model)

        # 3. 加载图片
        # 使用 Pillow 库打开图片
        img = PIL.Image.open(image_path)
        print("img", img)

        # 4. 构建提示词 (Prompt)
        # 我们可以给模型一个明确的指令，告诉它如何结合这两个输入
        prompt_instruction = "请结合这张图片的内容以及下方的文字描述，给出一个综合性的总结。"
        print("prompt_instruction", prompt_instruction)

        # 5. 发送请求
        # generate_content 接受一个列表，列表中可以包含文本字符串和图片对象
        response = model.generate_content([prompt_instruction, user_text, img])
        print("response", response)

        # 6. 返回结果
        return response.text

    except Exception as e:
        return f"发生错误: {e}"


# --- 调用示例 ---
if __name__ == "__main__":
    # 假设你有一张名为 sample_image.jpg 的图片
    # 请确保图片路径正确
    image_file = "test.png"

    # 用户提供的文字
    text_input = """这张图片是一个简单的卡通插画，描绘了一个包含车辆、自然元素和建筑的场景。

以下是图片中各个元素的详细描述：

中心主体： 图像的中心是一个巨大的棕色直角三角形，充当一个斜面（坡道）。一辆红色的卡通小汽车正沿着这个斜面向上行驶，车头朝向右上方。

背景与环境： 整个背景是晴朗的浅蓝色天空。底部有一条细细的绿色草地。

左侧元素：

在左上角，有一个明亮的黄色太阳，带有尖尖的光芒。

在左下角，太阳的下方，有一棵茂盛的绿色落叶树，有着棕色的树干。

右侧元素：

在右上角，有一只棕色的鸟正从带有绿叶的树枝上起飞。

在那只鸟的下方，有一只更小的蓝色蜂鸟在空中飞翔。

在右下角，有一座带有红屋顶、棕色门和蓝色窗户的小房子。

整体风格： 图像采用干净、扁平的矢量艺术风格，线条清晰，色彩鲜艳，类似儿童读物中的插图。这幅图经常被用来解释物理学中关于斜面、力和运动的基本概念。"""

    print("正在向 Gemini 发送请求...")

    # 执行函数
    if os.path.exists(image_file):
        summary = summarize_image_and_text(image_file, text_input)
        print("-" * 30)
        print("Gemini 的总结回复：\n")
        print(summary)
    else:
        print(f"找不到图片文件: {image_file}，请检查路径。")