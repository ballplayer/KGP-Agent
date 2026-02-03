import os
import json
import google.generativeai as genai
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ================= 配置区域 =================
# 1. 配置 API Key（从环境变量读取，避免硬编码）
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️ generateKG.py: 环境变量 GOOGLE_API_KEY 未设置。LLM/Embedding 调用将失败。请通过 PowerShell: $env:GOOGLE_API_KEY=\"<your_key>\"")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# 2. 配置网络代理 (如果你在国内，必须配置)
# 请根据你的代理软件修改端口，常见为 7890, 1080, 10809
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


# ================= 核心功能函数 =================

def extract_knowledge_graph(text_input):
    """
    调用 Gemini 模型，提取实体(含属性)和关系。
    """
    model = genai.GenerativeModel('gemini-2.5-flash')

    # Prompt 设计：核心是强制要求 JSON 结构，并包含 'attributes' 字段
    prompt = f"""
    任务：分析下面的文本，构建一个知识图谱。

    文本内容："{text_input}"

    要求：
    1. 提取文本中的**实体 (Entities)**。对于每个实体，请提取它的**名称 (id)**、**类型 (type)** 以及文本中提到的**属性 (attributes)**（以键值对形式）。
    2. 提取实体之间的**关系 (Relationships)**。
    3. 输出必须是合法的 JSON 格式，不要包含 Markdown 标记。

    JSON 输出模版：
    {{
        "entities": [
            {{
                "id": "实体名称", 
                "type": "实体类型(如人物、公司、时间)", 
                "attributes": {{ "职位": "CEO", "年龄": "50岁" }} 
            }}
        ],
        "relationships": [
            {{ "source": "实体1名称", "target": "实体2名称", "relation": "关系描述" }}
        ]
    }}
    """

    try:
        print("🤖 正在调用大模型进行分析...")
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}  # 强制 JSON
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"❌ API 调用或解析失败: {e}")
        return None


def visualize_graph(data):
    """
    绘制知识图谱
    """
    if not data: return

    G = nx.DiGraph()

    # 添加节点
    for entity in data.get("entities", []):
        # 将属性整合进节点信息中
        G.add_node(entity["id"], label=entity["type"], **entity.get("attributes", {}))

    # 添加边
    for rel in data.get("relationships", []):
        G.add_edge(rel["source"], rel["target"], label=rel["relation"])

    # --- 绘图配置 ---
    plt.figure(figsize=(12, 8))

    # 解决中文乱码问题 (Windows使用SimHei, Mac可能需要Arial Unicode MS)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 布局
    pos = nx.spring_layout(G, k=0.8, iterations=50)  # k值越大节点越分散

    # 画点
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.9)
    # 画点上的字
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # 画边
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray', arrowstyle='->', arrowsize=20)
    # 画边上的字
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.title("生成的知识图谱可视化")
    plt.axis('off')
    plt.show()


def save_data_to_files(data, prefix="kg_output"):
    """
    将提取的数据保存为 JSON, Excel, GraphML 三种格式
    """
    if not data: return

    print(f"\n💾 开始保存数据 (前缀: {prefix})...")

    # 1. 保存原始 JSON (最完整)
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"  - JSON 已保存: {prefix}.json")

    # 2. 保存为 Excel (拆分为 实体表 和 关系表)
    # 处理实体表（展平 attributes 字典）
    entities_list = []
    for ent in data["entities"]:
        base_info = {"id": ent["id"], "type": ent["type"]}
        # 将属性展开，例如 {"职位": "CEO"} 变成一列
        attributes = ent.get("attributes", {})
        full_info = {**base_info, **attributes}
        entities_list.append(full_info)

    df_entities = pd.DataFrame(entities_list)
    df_relations = pd.DataFrame(data["relationships"])

    with pd.ExcelWriter(f"{prefix}.xlsx") as writer:
        df_entities.to_excel(writer, sheet_name="实体(Entities)", index=False)
        df_relations.to_excel(writer, sheet_name="关系(Relationships)", index=False)
    print(f"  - Excel 已保存: {prefix}.xlsx (包含两个Sheet)")

    # 3. 保存为 GraphML (供 Gephi 等软件使用)
    G = nx.DiGraph()
    for ent in data["entities"]:
        # GraphML 的属性值必须是字符串、数字等简单类型，不能是字典
        # 这里简单处理，只存 type
        G.add_node(ent["id"], type=ent["type"])
    for rel in data["relationships"]:
        G.add_edge(rel["source"], rel["target"], label=rel["relation"])

    try:
        nx.write_graphml(G, f"{prefix}.graphml")
        print(f"  - GraphML 已保存: {prefix}.graphml")
    except Exception as e:
        print(f"  - GraphML 保存警告: {e} (可能是属性格式问题，已跳过)")


# ================= 主程序入口 =================
if __name__ == "__main__":
    # 测试文本
    input_text = """
这张图片是一个风格简洁、色彩鲜明的卡通插画，采用扁平矢量艺术风格，整体呈现出儿童读物般的视觉效果，常用于解释物理学中的基本概念，如斜面、力与运动。

画面中心是一个巨大的棕色直角三角形斜面（坡道），一辆红色的卡通小汽车正沿着这个斜面向上行驶，车头朝向右上方，展现了运动中的车辆。

背景是晴朗的浅蓝色天空，底部衬托着一条细细的绿色草地。

在画面的左侧，左上角高挂着一个明亮的黄色太阳，带有尖尖的光芒。太阳下方，左下角则矗立着一棵茂盛的绿色落叶树，有着结实的棕色树干。

画面的右侧也充满了生动的元素：右上角，一只棕色的鸟正从带有绿叶的树枝上起飞，展现出飞翔的姿态。在那只鸟的下方，一只更小的蓝色蜂鸟在空中轻盈地飞翔。最右下角，则是一座带有红色屋顶、棕色门和蓝色窗户的可爱小房子，为场景增添了一丝生活气息。

总而言之，这是一幅充满童趣和教育意义的插画，通过清晰的线条和鲜艳的色彩，描绘了一个包含自然风光、动物、建筑和车辆在斜面上运动的和谐场景。
    """

    print(f"📄 输入文本: \n{input_text.strip()}\n")

    # 1. 提取
    kg_data = extract_knowledge_graph(input_text)

    if kg_data:
        # 2. 打印预览
        print("\n🔍 提取结果预览:")
        print(json.dumps(kg_data, ensure_ascii=False, indent=2))

        # 3. 保存
        save_data_to_files(kg_data, prefix="KG_I")

        # 4. 可视化
        print("\n🎨 正在绘图...")
        visualize_graph(kg_data)
    else:
        print("程序结束：未提取到有效数据。")