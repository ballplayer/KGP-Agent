import os
import json
import google.generativeai as genai
import networkx as nx
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 1. 设置代理 (根据你的实际网络环境修改端口)
# 2. 配置 API Key
# 推荐：通过环境变量设置 API Key，避免在源码中明文保存
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️ 未检测到环境变量 GOOGLE_API_KEY，后续对外 API 调用可能会失败。请设置后重试。")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
# This demo is used to demonstrate the pruning process. Replace it with your own GOOGLE_API_KEY here
# 此demo用于演示剪枝过程，此处替换为自己的GOOGLE_API_KEY

# ================= 辅助函数：创建测试文件 =================
def create_dummy_file_if_not_exists(filename):
    """
    为了演示方便，如果本地没有 raw_kg.json，我们就创建一个。
    实际使用时，你可以直接使用你自己的文件。
    """
    if not os.path.exists(filename):
        dummy_data = {
            "entities": [
                {
                    "id": "移动式智能机器人原型",
                    "type": "产品原型",
                    "attributes": {
                        "定位": "未来智能家居概念",
                        "展示目的": "展示未来智能家居的概念",
                        "状态": "原型机,非已上市的成熟产品",
                        "展示场合": "科技展会",
                        "尺寸": "小型",
                        "移动方式": "轮式",
                        "能力": "具备一定自主移动能力",
                        "硬件平台状态": "功能完善但处于开发阶段",
                        "潜力": "智能助手或环境感知平台"
                    }
                },
                {"id": "未来智能家居概念", "type": "概念", "attributes": {}},
                {"id": "科技展会", "type": "场合", "attributes": {}}
            ],
            "relationships": [
                {"source": "移动式智能机器人原型", "target": "科技展会", "relation": "展示于"},
                {"source": "移动式智能机器人原型", "target": "未来智能家居概念", "relation": "定位为"},
                {"source": "移动式智能机器人原型", "target": "LiDAR（激光雷达）", "relation": "搭载"}
            ]
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, ensure_ascii=False, indent=4)
        print(f"ℹ️ 已自动创建测试文件: {filename}")


# ================= 核心逻辑函数 =================

def load_local_json(filepath):
    """读取本地 JSON 文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"❌ 错误：文件 {filepath} 格式不正确")
        return None


def prune_graph_with_gemini(input_data):
    """
    调用 Gemini 进行语义剪枝
    """
    model = genai.GenerativeModel('gemini-2.5-flash')

    # 将 JSON 转字符串作为 Prompt 上下文
    json_str = json.dumps(input_data, ensure_ascii=False, indent=2)

    prompt = f"""
    任务：对以下知识图谱数据进行“语义剪枝”，生成精简版图谱 (KG_P)。

    原始数据 JSON：
    {json_str}

    剪枝规则：
    1. **实体精简**：保留核心实体（如具体产品、关键组件），去除背景性或临时性的弱相关实体（如泛化的“场合”）。
    2. **属性去重**：合并含义重复的属性（例如“状态”和“硬件状态”），保留最关键的参数（如核心能力、定位），去除琐碎细节。
    3. **关系清理**：如果关系指向的实体被删除了，该关系也必须删除。

    输出要求：
    - 直接返回合法的 JSON 格式。
    - 不要使用 Markdown 代码块包裹。
    """

    print("🤖 正在请求大模型进行剪枝处理...")
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"❌ 大模型调用失败: {e}")
        return None


def visualize_kg(data, title="Pruned Knowledge Graph"):
    """
    可视化知识图谱
    """
    if not data: return

    G = nx.DiGraph()

    # 1. 构建图结构
    # 添加节点
    for entity in data.get("entities", []):
        # 节点标签显示 名字 + 类型
        node_label = f"{entity['id']}\n({entity['type']})"
        G.add_node(entity['id'], label=node_label, type=entity['type'])

    # 添加边
    for rel in data.get("relationships", []):
        G.add_edge(rel["source"], rel["target"], label=rel["relation"])

    # 2. 绘图设置
    plt.figure(figsize=(10, 8))

    # --- 字体设置 (防止中文乱码) ---
    # Windows 尝试 SimHei, Mac 尝试 Arial Unicode MS
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 布局算法
    pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)

    # 绘制节点
    # 区分核心节点颜色 (简单逻辑：度数高的节点颜色深)
    degrees = [val for (node, val) in G.degree()]
    nodes = nx.draw_networkx_nodes(G, pos,
                                   node_size=3500,
                                   node_color='lightgreen',
                                   alpha=0.9,
                                   edgecolors='gray')

    # 绘制节点文字
    # 这里我们只显示 ID，如果想显示属性，通常放在悬停提示里，但在静态图中不适合显示太多文字
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", font_family='sans-serif')

    # 绘制边
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='gray', arrowstyle='-|>', arrowsize=20)

    # 绘制边的文字
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5)

    plt.title(title, fontsize=15)
    plt.axis('off')

    # 保存图片
    plt.savefig("KG_P_visualization.png", format="png", dpi=300)
    print("🖼️ 可视化图片已保存为: KG_P_visualization.png")

    plt.show()


# ================= 主程序流程 =================
if __name__ == "__main__":
    # 1. 定义文件名
    input_file = "KG_I.json"
    output_file = "KG_P.json"

    # (可选) 如果你本地没有这个文件，这一步会自动生成一个供测试
    create_dummy_file_if_not_exists(input_file)

    # 2. 从本地读取
    print(f"📂 正在读取本地文件: {input_file}")
    raw_data = load_local_json(input_file)

    if raw_data:
        # 3. 执行剪枝
        pruned_data = prune_graph_with_gemini(raw_data)

        if pruned_data:
            # 4. 保存结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(pruned_data, f, ensure_ascii=False, indent=4)
            print(f"✅ 剪枝后的数据已保存至: {output_file}")

            # 打印对比
            print("\n--- 剪枝效果对比 ---")
            print(f"原始属性数 (第一个实体): {len(raw_data['entities'][0]['attributes'])}")
            try:
                print(f"剪枝后属性数 (第一个实体): {len(pruned_data['entities'][0]['attributes'])}")
            except:
                pass

            # 5. 可视化展示
            print("\n🎨 正在绘制可视化图谱...")
            visualize_kg(pruned_data)
        else:
            print("❌ 剪枝返回为空，流程终止。")
    else:
        print("❌ 读取文件失败，流程终止。")