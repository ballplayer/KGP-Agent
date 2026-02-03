import json
import os
import matplotlib.pyplot as plt
import numpy as np


# ================= 1. 辅助函数：生成模拟数据 =================
# 如果你本地没有这两个文件，运行此代码会自动生成，方便测试
def generate_dummy_files(file_i, file_p):
    if not os.path.exists(file_i) or not os.path.exists(file_p):
        print("ℹ️ 未找到本地文件，正在生成模拟数据...")

        # 模拟 剪枝前 (KG_I) - 数据比较冗余
        data_i = {
            "entities": [
                {"id": "A", "type": "产品", "attributes": {"k1": "v1", "k2": "v2", "k3": "v3"}},
                {"id": "B", "type": "组件", "attributes": {"k1": "v1"}},
                {"id": "C", "type": "背景", "attributes": {}},  # 将被剪枝
                {"id": "D", "type": "噪音", "attributes": {}}  # 将被剪枝
            ],
            "relationships": [
                {"source": "A", "target": "B", "relation": "包含"},
                {"source": "A", "target": "C", "relation": "展示于"},  # 将被剪枝
                {"source": "C", "target": "D", "relation": "关联"}  # 将被剪枝
            ]
        }

        # 模拟 剪枝后 (KG_P) - 数据精简
        data_p = {
            "entities": [
                {"id": "A", "type": "产品", "attributes": {"k1": "v1", "核心定位": "智能设备"}},  # 属性发生变化
                {"id": "B", "type": "组件", "attributes": {"k1": "v1"}}
            ],
            "relationships": [
                {"source": "A", "target": "B", "relation": "包含"}
            ]
        }

        with open(file_i, 'w', encoding='utf-8') as f: json.dump(data_i, f, indent=2)
        with open(file_p, 'w', encoding='utf-8') as f: json.dump(data_p, f, indent=2)


# ================= 2. 核心分析类 =================

class KGAnalyzer:
    def __init__(self, path_i, path_p):
        self.kg_i = self._load_json(path_i)
        self.kg_p = self._load_json(path_p)

    def _load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_stats(self, data):
        """计算单个图谱的基础统计数据"""
        entities = data.get("entities", [])
        relations = data.get("relationships", [])

        # 1. 实体数量
        num_entities = len(entities)

        # 2. 关系数量
        num_relations = len(relations)

        # 3. 属性总数 (累加所有实体的属性键值对数量)
        num_attributes = sum(len(e.get("attributes", {})) for e in entities)

        # 4. 获取实体ID集合 (用于后续集合运算)
        entity_ids = set(e["id"] for e in entities)

        return {
            "count_ent": num_entities,
            "count_rel": num_relations,
            "count_attr": num_attributes,
            "ids": entity_ids
        }

    def compare(self):
        """对比两个图谱"""
        stats_i = self.get_stats(self.kg_i)
        stats_p = self.get_stats(self.kg_p)

        # 计算差值
        diff_ent = stats_i["count_ent"] - stats_p["count_ent"]
        diff_rel = stats_i["count_rel"] - stats_p["count_rel"]
        diff_attr = stats_i["count_attr"] - stats_p["count_attr"]

        # 计算剪枝率 (Pruning Rate)
        rate_ent = (diff_ent / stats_i["count_ent"] * 100) if stats_i["count_ent"] else 0
        rate_rel = (diff_rel / stats_i["count_rel"] * 100) if stats_i["count_rel"] else 0
        rate_attr = (diff_attr / stats_i["count_attr"] * 100) if stats_i["count_attr"] else 0

        # 具体的集合差异
        removed_entities = stats_i["ids"] - stats_p["ids"]

        return {
            "before": stats_i,
            "after": stats_p,
            "diff": {
                "entities": diff_ent,
                "relations": diff_rel,
                "attributes": diff_attr
            },
            "rates": {
                "entities": rate_ent,
                "relations": rate_rel,
                "attributes": rate_attr
            },
            "details": {
                "removed_entity_ids": list(removed_entities)
            }
        }

    def visualize(self, result):
        """绘制对比图表"""
        labels = ['实体 (Entities)', '关系 (Relationships)', '属性条目 (Attributes)']
        before_vals = [result['before']['count_ent'], result['before']['count_rel'], result['before']['count_attr']]
        after_vals = [result['after']['count_ent'], result['after']['count_rel'], result['after']['count_attr']]

        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(10, 6))
        # 中文支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        rects1 = plt.bar(x - width / 2, before_vals, width, label='剪枝前 (KG_I)', color='#ff9999')
        rects2 = plt.bar(x + width / 2, after_vals, width, label='剪枝后 (KG_P)', color='#99ff99')

        # 添加标签
        plt.ylabel('数量')
        plt.title('知识图谱剪枝前后数据对比')
        plt.xticks(x, labels)
        plt.legend()

        # 在柱状图上方显示数值和下降百分比
        def autolabel(rects, is_after=False):
            for i, rect in enumerate(rects):
                height = rect.get_height()
                text = f'{height}'
                if is_after:
                    # 在“剪枝后”的柱子上显示下降百分比
                    rate_keys = ['entities', 'relations', 'attributes']
                    rate = result['rates'][rate_keys[i]]
                    text += f'\n(-{rate:.1f}%)'

                plt.annotate(text,
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2, is_after=True)

        plt.tight_layout()
        plt.show()


# ================= 3. 主程序 =================
if __name__ == "__main__":
    file_i = "KG_I.json"
    file_p = "KG_P.json"

    # 1. 准备数据
    generate_dummy_files(file_i, file_p)

    # 2. 执行分析
    print(f"📊 正在对比 {file_i} 和 {file_p} ...\n")
    analyzer = KGAnalyzer(file_i, file_p)
    res = analyzer.compare()

    # 3. 打印文本报告
    print("=" * 40)
    print(f"{'指标':<15} | {'剪枝前':<8} | {'剪枝后':<8} | {'减少量':<8} | {'剪枝率':<8}")
    print("-" * 60)

    metrics = [
        ("实体 (Node)", "count_ent", "entities"),
        ("关系 (Edge)", "count_rel", "relations"),
        ("属性 (Attr)", "count_attr", "attributes")
    ]

    for label, key_count, key_diff in metrics:
        val_i = res['before'][key_count]
        val_p = res['after'][key_count]
        diff = res['diff'][key_diff]
        rate = res['rates'][key_diff]
        print(f"{label:<15} | {val_i:<11} | {val_p:<11} | {diff:<11} | {rate:.1f}%")

    print("=" * 40)

    print("\n🗑️  被移除的实体 ID:")
    if res['details']['removed_entity_ids']:
        for eid in res['details']['removed_entity_ids']:
            print(f"   - {eid}")
    else:
        print("   (无实体被完全移除)")

    # 4. 可视化
    print("\n🎨 正在生成对比图表...")
    analyzer.visualize(res)