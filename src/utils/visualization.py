import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from typing import List, Dict


def plot_propagation_history(
    propagation_results: List[Dict], title: str = "propagation process"
):
    """
    绘制传播历史曲线

    参数:
        propagation_results: 传播结果列表
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))

    # 计算平均传播历史
    max_steps = max(len(result["activation_history"]) for result in propagation_results)
    mean_history = np.zeros(max_steps)

    for result in propagation_results:
        history = result["activation_history"]
        mean_history[: len(history)] += history
        plt.plot(history, alpha=0.1, color="gray")

    mean_history /= len(propagation_results)
    plt.plot(mean_history, "b-", linewidth=2, label="average propagation curve")

    plt.title(title)
    plt.xlabel("propagation steps")
    plt.ylabel("activated nodes")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_node_importance(
    graph: nx.Graph,
    importance_scores: np.ndarray,
    title: str = "node importance distribution",
):
    """
    可视化节点重要性分布

    参数:
        graph: NetworkX图对象
        importance_scores: 节点重要性分数
        title: 图表标题
    """
    plt.figure(figsize=(12, 8))

    # 创建布局
    pos = nx.spring_layout(graph, seed=42)

    # 绘制节点
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=importance_scores,
        node_size=100,
        cmap=plt.cm.viridis,
        alpha=0.8,
    )

    # 绘制边
    nx.draw_networkx_edges(graph, pos, alpha=0.2)

    # 添加颜色条
    plt.colorbar(nodes, label="node importance")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
