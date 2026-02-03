import os
import json
import time
import webbrowser
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai

# --- 导入你的五个模块 ---
# 必须确保 api.py, generateKG.py, pruningKG.py, compareKG.py, KG2Image.py 在同一目录下
try:
    import api
    import generateKG
    import pruningKG
    import compareKG
    import KG2Image
    import match_kb
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保所有 5 个 python 文件都在当前目录下。")
    exit(1)

# ================= 全局配置 =================
# 1. 配置 API Key（从环境变量读取）
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️ workflow.py: 环境变量 GOOGLE_API_KEY 未设置。请在 PowerShell 中设置：$env:GOOGLE_API_KEY=\"<your_key>\"")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

PROXY_URL = "http://127.0.0.1:7890"

os.environ["HTTP_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = PROXY_URL

# ================= 核心：劫持 plt.show 实现“先保存后弹窗” =================
# 全局变量，用于控制当前图片应该保存到哪里
current_save_path = None
original_show = plt.show


def custom_show(*args, **kwargs):
    """
    自定义的显示函数：
    1. 先把图保存到 current_save_path
    2. 再调用原始的 plt.show() 弹出窗口
    """
    global current_save_path
    if current_save_path:
        # 保存图片
        plt.savefig(current_save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 [自动保存] 图片已保存至: {current_save_path}")
        print("   👀 [本地显示] 请查看弹出的窗口 (关闭窗口后程序将继续)...")

    # 弹出窗口 (这行代码是阻塞的，关闭窗口后才会继续往下走)
    original_show(*args, **kwargs)


# 将 matplotlib 的 show 替换为我们的自定义函数
plt.show = custom_show


# ================= 流程工具函数 =================

def create_output_folder():
    """创建带时间戳的输出目录"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"Output_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    print(f"📂 创建输出文件夹: {os.path.abspath(folder_name)}")
    return folder_name


def main():
    global current_save_path
    print("🚀 启动本地可视化流水线 (Local Show Mode)...\n")

    # 0. 准备工作
    output_dir = create_output_folder()

    # 设定输入
    input_image_path = "test.png"
    user_prompt_text = "请详细描述这张图片中的所有元素、它们的位置关系以及颜色特征。"

    if not os.path.exists(input_image_path):
        print(f"❌ 错误: 找不到输入图片 {input_image_path}，请准备一张 test.png。")
        return

    # ================= 步骤 1: API (图片总结) =================
    print("\n--- [Step 1/5] 生成图片总结 (api.py) ---")
    summary_text = api.summarize_image_and_text(input_image_path, user_prompt_text)

    if not summary_text or "发生错误" in summary_text:
        print("❌ 步骤 1 失败。")
        return

    # 保存 TXT
    txt_path = os.path.join(output_dir, "1_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"✅ 总结文字已保存: {txt_path}")
    print(f"📝 内容预览: {summary_text[:50]}...")

    # ================= 步骤 2: GenerateKG (初始图谱) =================
    print("\n--- [Step 2/5] 生成初始知识图谱 (generateKG.py) ---")
    kg_i_data = generateKG.extract_knowledge_graph(summary_text)

    if not kg_i_data:
        print("❌ 步骤 2 失败。")
        return

    # 保存 JSON
    kg_i_json_path = os.path.join(output_dir, "2_KG_I.json")
    with open(kg_i_json_path, "w", encoding="utf-8") as f:
        json.dump(kg_i_data, f, ensure_ascii=False, indent=4)
    print(f"✅ JSON 已保存: {kg_i_json_path}")

    # 设置保存路径 -> 调用绘图 -> 触发 custom_show -> 保存并弹窗
    print("   🎨 正在绘制初始图谱...")
    current_save_path = os.path.join(output_dir, "2_KG_I.png")
    # 调用原子模块的绘图函数 (它内部会调用 plt.show，被我们劫持了)
    generateKG.visualize_graph(kg_i_data)

    # ================= 步骤 3: PruningKG (图谱剪枝) =================
    print("\n--- [Step 3/5] 进行语义剪枝 (pruningKG.py) ---")
    kg_p_data = pruningKG.prune_graph_with_gemini(kg_i_data)

    if not kg_p_data:
        print("❌ 步骤 3 失败。")
        return

    # 保存 JSON
    kg_p_json_path = os.path.join(output_dir, "3_KG_P.json")
    with open(kg_p_json_path, "w", encoding="utf-8") as f:
        json.dump(kg_p_data, f, ensure_ascii=False, indent=4)
    print(f"✅ JSON 已保存: {kg_p_json_path}")

    # 设置保存路径 -> 调用绘图 -> 触发 custom_show -> 保存并弹窗
    print("   🎨 正在绘制剪枝图谱...")
    current_save_path = os.path.join(output_dir, "3_KG_P.png")
    pruningKG.visualize_kg(kg_p_data, title="Pruned Knowledge Graph")

    # ================= 步骤 4: CompareKG (对比分析) =================
    print("\n--- [Step 4/5] 对比分析 (compareKG.py) ---")

    analyzer = compareKG.KGAnalyzer(kg_i_json_path, kg_p_json_path)
    results = analyzer.compare()

    # 保存 Excel
    excel_path = os.path.join(output_dir, "4_analysis_data.xlsx")
    metrics_data = {
        "指标": ["实体数量", "关系数量", "属性总数"],
        "KG_I": [results['before']['count_ent'], results['before']['count_rel'], results['before']['count_attr']],
        "KG_P": [results['after']['count_ent'], results['after']['count_rel'], results['after']['count_attr']],
        "减少量": [results['diff']['entities'], results['diff']['relations'], results['diff']['attributes']]
    }
    pd.DataFrame(metrics_data).to_excel(excel_path, index=False)
    print(f"✅ 数据已保存 Excel: {excel_path}")

    # 设置保存路径 -> 调用绘图 -> 触发 custom_show -> 保存并弹窗
    print("   🎨 正在生成对比图表...")
    current_save_path = os.path.join(output_dir, "4_analysis_chart.png")
    analyzer.visualize(results)

    # ================= 步骤 5: KG2Image (SVG 场景绘制) =================
    print("\n--- [Step 5/5] 绘制 SVG 插画 (KG2Image.py) ---")

    svg_content = KG2Image.generate_generic_scene_svg(kg_p_data)

    if svg_content:
        svg_path = os.path.join(output_dir, "5_scene.svg")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        print(f"✅ SVG 已保存: {svg_path}")

        # 本地 Show SVG：SVG文件没有通用的弹窗查看器，最好的方法是调用浏览器打开
        print("   👀 [本地显示] 正在调用浏览器打开 SVG 图片...")
        webbrowser.open('file://' + os.path.abspath(svg_path))
    else:
        print("❌ SVG 生成失败。")

    # ================= 步骤 6: 匹配考试知识点 (match_kb.py) =================
    print("\n--- [Step 6/6] 匹配考试知识点 (match_kb.py) ---")
    try:
        # match_kb.match_and_rank expects (kg_path, kb_path, output_dir)
        match_res = match_kb.match_and_rank(kg_p_json_path, 'exam_kb.json', output_dir)
        if match_res and match_res.get('selected'):
            print("✅ 已识别核心知识点:")
            for s in match_res['selected']:
                print(f"   - {s['title']} (score={s['combined_score']:.3f})")
        else:
            print("⚠️ 未能识别明确的核心知识点，请检查输出文件或扩充 KB。")
    except Exception as e:
        print(f"❌ 匹配考试知识点失败: {e}")

    print(f"\n🎉 全流程执行完毕！文件保存在: {output_dir}")


if __name__ == "__main__":
    main()