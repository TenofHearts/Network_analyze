import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from typing import List, Dict


def plot_propagation_history(
    propagation_results: List[Dict],
    title: str = "propagation process",
    save_path: str = None,
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
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_node_importance(
    graph: nx.Graph,
    importance_scores: np.ndarray,
    title: str = "node importance distribution",
    save_dir: str = None,
):
    """
    可视化节点重要性分布。当节点数小于10000时使用networkx直接可视化，
    否则导出为csv文件以供Gephi可视化。

    参数:
        graph: NetworkX图对象
        importance_scores: 节点重要性分数
        title: 图表标题
        save_dir: 保存csv文件的目录路径
    """
    print(f"原始图包含 {len(graph)} 个节点和 {graph.number_of_edges()} 条边")

    if len(graph) <= 10000:
        # 直接使用networkx可视化
        plt.figure(figsize=(12, 8))
        print("计算节点布局...")
        pos = nx.spring_layout(graph, seed=42, k=1 / np.sqrt(len(graph)))

        print("绘制节点...")
        nodes = nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=importance_scores,
            node_size=5,
            cmap=plt.cm.viridis,
            alpha=0.8,
        )

        print("绘制边...")
        nx.draw_networkx_edges(graph, pos, alpha=0.1, width=0.5)

        plt.colorbar(nodes, label="node importance")
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        print("图形绘制完成")
        if save_dir is not None:
            plt.savefig(save_dir / f"{title}_importance.png")
        else:
            plt.show()
        plt.close()  # 关闭图形，避免内存泄漏

    else:
        # 导出为csv文件
        from pathlib import Path
        import pandas as pd

        if save_dir is None:
            raise ValueError("需要提供save_dir参数以保存csv文件")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 准备节点数据
        nodes_data = pd.DataFrame(
            {
                "name": list(graph.nodes()),
                "importance": importance_scores,
                "shared name": list(graph.nodes()),  # Cytoscape需要shared name列
            }
        )

        # 准备边数据
        edges_data = pd.DataFrame(
            {
                "source": [e[0] for e in graph.edges()],
                "target": [e[1] for e in graph.edges()],
                "interaction": ["interacts"]
                * len(graph.edges()),  # Cytoscape需要interaction列
                "shared name": [
                    f"{e[0]} (interacts) {e[1]}" for e in graph.edges()
                ],  # 边的唯一标识
            }
        )

        # 保存文件
        nodes_file = save_path / "nodes.csv"
        edges_file = save_path / "edges.csv"

        nodes_data.to_csv(nodes_file, index=False)
        edges_data.to_csv(edges_file, index=False)

        print(f"节点数据已保存至: {nodes_file}")
        print(f"边数据已保存至: {edges_file}")
        print("请使用Cytoscape软件打开这些文件进行可视化\n")
