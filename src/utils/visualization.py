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

    性能分析:
    1. 布局计算最耗时 - nx.kamada_kawai_layout() 需要O(N^2)时间复杂度
    2. 边的绘制次之 - 需要O(E)时间处理所有边
    3. 节点绘制较快 - 一次性绘制所有节点O(N)
    4. 其他操作(如数组复制等)耗时很小
    """
    plt.figure(figsize=(12, 8))

    # 使用更快的布局算法
    pos = nx.kamada_kawai_layout(graph)

    # 预先计算节点大小以避免重复计算
    n_nodes = len(graph)
    node_sizes = np.ones(n_nodes)

    # 使用numpy数组存储颜色值以提高效率
    node_colors = importance_scores.copy()

    # 一次性绘制所有节点
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.viridis,
        alpha=0.6,
    )

    # 一次性绘制所有边,使用稀疏矩阵存储边信息
    edge_list = list(graph.edges())
    if edge_list:
        edge_pos = np.array([(pos[e[0]], pos[e[1]]) for e in edge_list])
        plt.plot(
            edge_pos[:, :, 0].T,
            edge_pos[:, :, 1].T,
            "-",
            color="gray",
            alpha=0.2,
            linewidth=0.5,
        )

    plt.colorbar(nodes, label="node importance")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
