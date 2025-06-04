import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.propagation import IndependentCascade, LinearThreshold
import os


def plot_propagation_curve(
    activation_history: List[int],
    title: str = "Propagation Curve",
    save_path: str = None,
) -> None:
    """
    绘制传播过程中激活节点数量的变化曲线

    参数:
        activation_history: 每轮激活节点数量的历史记录
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # 绘制曲线
    plt.plot(
        range(len(activation_history)),
        activation_history,
        marker="o",
        linestyle="-",
        color="#2ecc71",
        linewidth=2,
        markersize=6,
    )

    # 设置标题和标签
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Propagation Round", fontsize=12)
    plt.ylabel("Activated Nodes", fontsize=12)

    # 设置网格
    plt.grid(True, linestyle="--", alpha=0.7)

    # 美化
    sns.despine()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_propagation_network(
    graph: nx.Graph,
    seed_nodes: Set[int],
    activation_times: Dict[int, int],
    infection_path: Dict[int, int],
    title: str = "Propagation Network",
    save_path: str = None,
    node_size: int = 5,
    seed_node_size: int = 20,
    pos: Dict = None,
) -> None:
    """
    可视化传播网络，展示感染过程和路径

    参数:
        graph: 网络图
        seed_nodes: 种子节点集合
        activation_times: 节点激活时间字典
        infection_path: 感染路径字典
        title: 图表标题
        save_path: 保存路径
        node_size: 普通节点大小
        seed_node_size: 种子节点大小
        pos: 预计算的节点位置字典
    """
    fig, ax = plt.subplots(figsize=(15, 15))

    # 创建只包含被感染节点的子图
    infected_nodes = set(activation_times.keys())
    subgraph = graph.subgraph(infected_nodes).copy()

    # 创建自定义颜色映射
    max_time = max(activation_times.values())
    if max_time == 0:  # 如果所有节点都是种子节点
        colors = plt.cm.viridis(np.linspace(0, 1, 2))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, max_time + 1))
    cmap = LinearSegmentedColormap.from_list("custom_viridis", colors)

    # 设置节点颜色
    node_colors = []
    for node in subgraph.nodes():
        if node in seed_nodes:
            node_colors.append("#e74c3c")  # 种子节点使用红色
        else:
            time = activation_times[node]
            node_colors.append(cmap(time / max_time if max_time > 0 else 0))

    # 设置节点大小
    node_sizes = []
    for node in subgraph.nodes():
        if node in seed_nodes:
            node_sizes.append(seed_node_size)
        else:
            node_sizes.append(node_size)

    # 使用预计算的位置或计算新位置
    if pos is None:
        pos = nx.random_layout(graph)  # 使用最快的布局算法

    # 绘制边
    nx.draw_networkx_edges(
        subgraph, pos, alpha=0.2, width=0.5, edge_color="gray", ax=ax
    )

    # 绘制节点
    nx.draw_networkx_nodes(
        subgraph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.6, ax=ax
    )

    # 添加颜色条
    if max_time > 0:  # 只在有非种子节点时添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_time))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Infection Round", fontsize=12)

    # 设置标题
    ax.set_title(title, fontsize=14, pad=20)

    # 移除坐标轴
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return pos


def visualize_propagation(
    graph: nx.Graph,
    propagation_result: Dict,
    seed_nodes: Set[int],
    title_prefix: str = "",
    save_dir: str = None,
    pos: Dict = None,
) -> None:
    """
    可视化完整的传播过程

    参数:
        graph: 网络图
        propagation_result: 传播结果字典
        seed_nodes: 种子节点集合
        title_prefix: 图表标题前缀
        save_dir: 保存目录，如果为None则显示图表
    """
    # 绘制传播曲线
    curve_title = f"{title_prefix}Propagation Curve"
    curve_path = f"{save_dir}/propagation_curve.png" if save_dir else None
    plot_propagation_curve(
        propagation_result["activation_history"],
        title=curve_title,
        save_path=curve_path,
    )

    # 绘制传播网络
    network_title = f"{title_prefix}Propagation Network"
    network_path = f"{save_dir}/propagation_network.png" if save_dir else None
    plot_propagation_network(
        graph=graph,
        seed_nodes=seed_nodes,
        activation_times=propagation_result["activation_times"],
        infection_path=propagation_result["infection_path"],
        title=network_title,
        save_path=network_path,
        pos=pos,  # 传入预计算的位置
    )


def visualize_propagation_example(
    graph: nx.Graph,
    seed_nodes: Set[int],
    model_type: str = "ic",
    max_steps: int = 10,
    save_dir: str = None,
    pos: Dict = None,
) -> None:
    """
    传播可视化的示例函数，可以直接调用进行可视化

    参数:
        graph: 网络图
        seed_nodes: 种子节点集合
        model_type: 传播模型类型，可选 "ic" 或 "lt"
        max_steps: 最大传播步数
        save_dir: 保存目录，如果为None则显示图表
        pos: 预计算的节点位置字典
    """
    # 选择传播模型
    if model_type.lower() == "lt":
        propagation = LinearThreshold(graph)
    else:
        propagation = IndependentCascade(graph)

    # 进行传播模拟
    result = propagation.simulate(list(seed_nodes), max_steps=max_steps)

    # 可视化传播过程
    model_name = (
        "Independent Cascade" if model_type.lower() == "ic" else "Linear Threshold"
    )
    visualize_propagation(
        graph=graph,
        propagation_result=result,
        seed_nodes=seed_nodes,
        title_prefix=f"{model_name} - ",
        save_dir=save_dir,
        pos=pos,  # 传入预计算的位置
    )


def visualize_propagation_comparison(
    graph: nx.Graph,
    seed_nodes: Set[int],
    max_steps: int = 5,
    save_dir: str = None,
    pos: Dict = None,
) -> None:
    """
    比较两种传播模型的可视化结果

    参数:
        graph: 网络图
        seed_nodes: 种子节点集合
        max_steps: 最大传播步数
        save_dir: 保存目录，如果为None则显示图表
    """
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ic_dir = f"{save_dir}/ic_model"
        lt_dir = f"{save_dir}/lt_model"
        os.makedirs(ic_dir, exist_ok=True)
        os.makedirs(lt_dir, exist_ok=True)
    else:
        ic_dir = None
        lt_dir = None

    # 可视化IC模型
    print("可视化独立级联模型传播过程...")
    visualize_propagation_example(
        graph=graph,
        seed_nodes=seed_nodes,
        model_type="ic",
        max_steps=max_steps,
        save_dir=ic_dir,
        pos=pos,  # 传入预计算的位置
    )

    # 可视化LT模型
    print("可视化线性阈值模型传播过程...")
    visualize_propagation_example(
        graph=graph,
        seed_nodes=seed_nodes,
        model_type="lt",
        max_steps=max_steps,
        save_dir=lt_dir,
        pos=pos,  # 传入预计算的位置
    )


if __name__ == "__main__":
    from test_main import load_network, betweenness_selection, gat_selection

    # 加载网络数据
    print("加载网络数据...")
    graph, features = load_network("data/processed/facebook_processed.pkl")

    # 设置GAT模型路径
    gat_model_paths = [
        "outputs/twitter/model/twitter_gat_model.pt",
        "outputs/epinions/model/epinions_gat_model.pt",
        "outputs/facebook/model/facebook_gat_model.pt",
    ]

    # 定义算法
    algorithms = {
        "Betweeness": betweenness_selection,
    }

    # 添加GAT算法
    for model_path in gat_model_paths:
        algorithms[f"GAT_{model_path.split('/')[-1].split('.')[0]}"] = (
            lambda g, k, m=model_path: gat_selection(g, k, features, m)
        )

    # 计算全局布局
    print("计算全局布局...")
    pos = nx.spectral_layout(graph)

    # 为每个算法生成可视化结果
    for name, algorithm in algorithms.items():
        print(f"\n使用 {name} 算法选择种子节点...")
        seed_nodes = algorithm(graph, 400)

        print(f"可视化 {name} 算法的传播过程...")
        visualize_propagation_comparison(
            graph=graph,
            seed_nodes=seed_nodes,
            max_steps=3,
            save_dir=f"visualizations/{name}",
        )
