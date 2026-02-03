import os
import json
import time
import numpy as np
import networkx as nx
import google.generativeai as genai

# ================= 配置区域 =================
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
# 请替换为你的实际 API Key
# 使用环境变量读取 API Key，避免在代码中硬编码密钥
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️ 环境变量 GOOGLE_API_KEY 未设置。LLM/Embedding 调用将会失败。\n   请在运行前通过 PowerShell: $env:GOOGLE_API_KEY=\"<your_key>\" 设置。")
else:
    genai.configure(api_key=GOOGLE_API_KEY)


class HybridPruner:
    def __init__(self):
        self.model_llm = genai.GenerativeModel('gemini-2.5-flash')
        # 使用 embedding 模型进行向量化
        self.model_embed = "models/text-embedding-004"

    def _calculate_cosine_similarity(self, vec_a, vec_b):
        """计算两个向量的余弦相似度"""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    def step_1_topology_pruning(self, data):
        """
        策略1: 拓扑结构剪枝
        - 利用 NetworkX 构建图。
        - 删除孤立节点 (Degree=0)。
        - 删除非核心连通分量 (如果图不仅是一个整体)。
        - 规则过滤: 删除在 Schema 黑名单中的类型 (如 '背景', '噪音')。
        """
        print("   ✂️ [Step 1] 执行拓扑结构剪枝...")

        entities = {e['id']: e for e in data['entities']}
        relations = data['relationships']

        # 1.1 构建图
        G = nx.Graph()  # 使用无向图计算连通性
        for e_id in entities.keys():
            G.add_node(e_id)
        for r in relations:
            G.add_edge(r['source'], r['target'])

        initial_count = len(G.nodes)

        # 1.2 移除孤立节点 (Degree = 0)
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)

        # 1.3 类型黑名单过滤 (模拟 Schema 约束)
        # 假设我们需要移除这些类型的实体，除非它们是连接度很高的枢纽
        blacklist_types = ["背景", "噪音", "环境描述", "修饰词"]
        nodes_to_remove = []
        for node in G.nodes:
            if node in entities:
                e_type = entities[node]['type']
                degree = G.degree[node]
                # 如果是黑名单类型，且不是核心枢纽(度数<=1)，则删除
                if any(bt in e_type for bt in blacklist_types) and degree <= 1:
                    nodes_to_remove.append(node)

        G.remove_nodes_from(nodes_to_remove)

        # 1.4 重建数据
        valid_nodes = set(G.nodes)
        new_entities = [e for e in data['entities'] if e['id'] in valid_nodes]
        new_relations = [r for r in data['relationships']
                         if r['source'] in valid_nodes and r['target'] in valid_nodes]

        print(f"      - 移除了 {initial_count - len(valid_nodes)} 个拓扑冗余/孤立实体。")
        return {"entities": new_entities, "relationships": new_relations}

    def step_2_semantic_fusion(self, data):
        """
        策略2: 基于向量嵌入的属性融合
        - 对同一个实体的所有属性进行 Embedding。
        - 计算两两相似度，如果 > 0.9 则合并。
        """
        print("   🧬 [Step 2] 执行向量语义融合 (属性去重)...")

        new_entities = []

        for entity in data['entities']:
            attrs = entity.get('attributes', {})
            if len(attrs) < 2:
                new_entities.append(entity)
                continue

            keys = list(attrs.keys())
            # 构造语义文本: "属性名: 属性值"
            texts = [f"{k}: {attrs[k]}" for k in keys]

            try:
                # 批量获取向量 (Gemini Embedding API)
                embeddings = genai.embed_content(
                    model=self.model_embed,
                    content=texts,
                    task_type="semantic_similarity"
                )['embedding']

                # 标记需要删除的冗余属性 Key
                keys_to_remove = set()

                for i in range(len(keys)):
                    if keys[i] in keys_to_remove: continue
                    for j in range(i + 1, len(keys)):
                        if keys[j] in keys_to_remove: continue

                        # 计算相似度
                        sim = self._calculate_cosine_similarity(embeddings[i], embeddings[j])

                        # 阈值判定 (0.9 表示非常相似)
                        if sim > 0.9:
                            # 保留逻辑: 保留描述更短的(更精简)，或者保留特定关键词
                            # 这里简单演示：保留 i，删除 j
                            keys_to_remove.add(keys[j])
                            # print(f"      - [融合] '{texts[j]}' (冗余) -> 被 '{texts[i]}' 覆盖 (相似度: {sim:.2f})")

                # 重建属性字典
                new_attrs = {k: v for k, v in attrs.items() if k not in keys_to_remove}
                entity['attributes'] = new_attrs

            except Exception as e:
                print(f"      ⚠️ 向量计算出错 (可能是配额限制): {e}")
                pass

            new_entities.append(entity)

        data['entities'] = new_entities
        return data

    def step_3_llm_refinement(self, data):
        """
        策略3: LLM 最终精修
        - 将清洗过的数据发给 LLM，做最后的逻辑判断。
        - 此时 Token 消耗已大幅降低。
        """
        print("   🧠 [Step 3] 执行 LLM 最终逻辑精修...")

        json_str = json.dumps(data, ensure_ascii=False, indent=2)

        prompt = f"""
        你是一个知识图谱专家。上述数据已经经过了初步的拓扑清洗和向量去重。
        现在的任务是进行最后的**逻辑剪枝**。

        输入数据:
        {json_str}

        请执行以下操作：
        1. **检查关系合理性**: 删除逻辑上不通顺或错误的关系。
        2. **实体合并**: 如果图谱中存在两个不同 ID 但指代同一事物的实体（例如 "CEO" 和 "马斯克" 如果分开节点了），请将它们合并。
        3. **格式规范化**: 确保输出标准的 JSON。

        请直接返回优化后的 JSON 数据，不要包含 Markdown 标记。
        """

        try:
            response = self.model_llm.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"      ❌ LLM 调用失败: {e}")
            return data  # 如果失败，返回上一步的结果

    def run(self, input_data):
        """流水线入口"""
        # Step 1: 拓扑清洗
        data_s1 = self.step_1_topology_pruning(input_data)

        # Step 2: 向量融合
        data_s2 = self.step_2_semantic_fusion(data_s1)

        # Step 3: LLM 最终确认
        data_final = self.step_3_llm_refinement(data_s2)

        return data_final


# ================= 兼容旧版接口 =================
# 为了让 main_pipeline.py 能直接调用，保持函数名一致
def prune_graph_with_gemini(input_data):
    pruner = HybridPruner()
    return pruner.run(input_data)


# ================= 可视化函数 (保持不变) =================
def visualize_kg(data, title="Optimized Pruned Graph"):
    if not data: return
    G = nx.DiGraph()
    for entity in data.get("entities", []):
        node_label = f"{entity['id']}\n({entity['type']})"
        G.add_node(entity['id'], label=node_label, type=entity['type'])
    for rel in data.get("relationships", []):
        G.add_edge(rel["source"], rel["target"], label=rel["relation"])

    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=3500, node_color='lightgreen', alpha=0.9, edgecolors='gray')
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", font_family='sans-serif')
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='gray', arrowstyle='-|>', arrowsize=20)
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5)

    plt.title(title, fontsize=15)
    plt.axis('off')
    plt.savefig("KG_P_visualization.png", format="png", dpi=300)
    plt.show()


# ================= 单元测试 =================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 模拟数据：包含冗余属性和孤立点
    dummy_data = {
        "entities": [
            {
                "id": "机器人",
                "type": "产品",
                "attributes": {
                    "状态": "开发中",
                    "开发进度": "正在研发",  # 语义重复
                    "颜色": "白色"
                }
            },
            {"id": "噪音点A", "type": "背景", "attributes": {}},  # 孤立点
            {"id": "展会", "type": "场合", "attributes": {}}
        ],
        "relationships": [
            {"source": "机器人", "target": "展会", "relation": "展示于"}
        ]
    }

    print("--- 开始测试优化版剪枝 ---")
    optimized_kg = prune_graph_with_gemini(dummy_data)

    print("\n--- 结果 ---")
    print(json.dumps(optimized_kg, ensure_ascii=False, indent=2))