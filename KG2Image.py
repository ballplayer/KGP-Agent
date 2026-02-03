import os
import json
import re
import google.generativeai as genai

# ================= 配置区域 =================
# 1. 设置代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 2. API Key（从环境变量读取，避免硬编码）
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️ KG2Image.py: 环境变量 GOOGLE_API_KEY 未设置。SVG 生成将失败，请设置 $env:GOOGLE_API_KEY=\"<your_key>\"")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# ================= 功能函数 =================

def load_json_data(filepath):
    """读取本地 JSON 文件"""
    if not os.path.exists(filepath):
        print(f"❌ 错误：找不到文件 {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_generic_scene_svg(json_data):
    """
    通用版：不预设具体物体，让模型根据 JSON 内容自动发挥
    """
    model = genai.GenerativeModel('gemini-2.5-flash')

    # 将 JSON 数据转为字符串
    data_str = json.dumps(json_data, ensure_ascii=False, indent=2)

    # --- 核心通用 Prompt ---
    prompt = f"""
    You are an expert **SVG Illustrator** capable of visualizing any structured data into a scene.

    **Your Task:** Analyze the provided JSON data, understand the entities and their attributes, and draw a **flat-style cartoon SVG illustration** that represents the scene described by the data.

    **Input Data:**
    {data_str}

    **Step-by-Step Instructions (Do not hardcode specific objects, derive them from data):**

    1.  **Identify the Subject (Visual Analysis)**:
        -   Look for the central or most detailed entity in the `entities` list. This is your **Main Subject**.
        -   Analyze its `type` and `id`. (e.g., If it's a "Cat", draw a cat. If it's a "Car", draw a car).
        -   Analyze its `attributes`. Use these to determine color, shape, size, and accessories. (e.g., "color: red" -> fill red; "mood: happy" -> draw a smile).

    2.  **Identify the Context (Background)**:
        -   Look for other entities connected to the subject (via `relationships`).
        -   Use these to draw the **Background** or **Props**. (e.g., If relation is "located in Forest", draw trees in the background).

    3.  **Artistic Style**:
        -   **Style**: Flat Design, Minimalist, Vector Illustration.
        -   **Colors**: Use a harmonious color palette suitable for the subject.
        -   **Composition**: Center the Main Subject.
        -   **Canvas**: viewBox="0 0 512 512".

    4.  **Output Format**:
        -   Provide **ONLY** the raw SVG code.
        -   Do not include markdown code blocks (like ```svg).
        -   Do not include text explanations.

    **Now, interpret the JSON and generate the SVG code.**
    """

    print("🎨 Gemini 正在分析数据并构思画面 (通用模式)...")

    try:
        response = model.generate_content(prompt)
        content = response.text

        # --- 清洗数据 ---
        # 提取 <svg>...</svg>
        match = re.search(r'(<svg[\s\S]*?</svg>)', content)
        if match:
            svg_code = match.group(1)
        else:
            # 兜底清洗
            svg_code = content.replace("```xml", "").replace("```svg", "").replace("```", "").strip()

        return svg_code

    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return None


def save_svg(svg_code, filename):
    if not svg_code or "<svg" not in svg_code:
        print("❌ 生成的内容无效，跳过保存。")
        return

    with open(filename, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"✅ 插画已保存至: {os.path.abspath(filename)}")


# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 输入文件 (可以是机器人，也可以是任何其他东西)
    input_file = "KG_P.json"
    output_file = "generic_scene.svg"

    # 2. 读取
    data = load_json_data(input_file)

    if data:
        # 3. 生成
        svg_content = generate_generic_scene_svg(data)

        # 4. 保存
        save_svg(svg_content, output_file)