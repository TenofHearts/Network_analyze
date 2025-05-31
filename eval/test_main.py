import networkx as nx
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from eval.test_metrics import evaluate_model_performance
import time
from typing import List, Set, Dict, Tuple
import torch
import os
import pickle

from src.models.gat import GATModel
from src.data.dataset import SNAPDataset


def load_network(file_path: str) -> Tuple[nx.Graph, np.ndarray]:
    """加载网络数据和特征

    参数:
        file_path: 数据文件路径

    返回:
        graph: 网络图
        features: 节点特征矩阵
    """
    if file_path.endswith(".pkl"):
        # 加载预处理数据
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            # 从缓存数据中获取图数据
            dataset = SNAPDataset(
                data["dataset_name"], n_simulations=data["n_simulations"]
            )
            graph = dataset.load_graph()
            return graph, data["features"]
    else:
        # 加载原始边列表数据
        graph = nx.read_edgelist(file_path, create_using=nx.Graph(), nodetype=int)
        return graph, None


def degree_centrality_selection(graph: nx.Graph, k: int) -> Set[int]:
    """基于度中心性的关键节点选择"""
    degrees = nx.degree_centrality(graph)
    # 只返回节点ID，不返回分数
    return {
        node
        for node, _ in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:k]
    }


def pagerank_selection(graph: nx.Graph, k: int) -> Set[int]:
    """基于PageRank的关键节点选择"""
    pagerank = nx.pagerank(graph)
    # 只返回节点ID，不返回分数
    return {
        node
        for node, _ in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:k]
    }


def betweenness_selection(graph: nx.Graph, k: int) -> Set[int]:
    """基于介数中心性的关键节点选择（使用近似算法）"""
    # 使用近似算法，采样k*10个节点
    betweenness = nx.betweenness_centrality(
        graph,
        k=min(k * 10, len(graph)),  # 采样节点数
        seed=42,  # 固定随机种子以保证结果可复现
    )
    return {
        node
        for node, _ in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:k]
    }


def gat_selection(
    graph: nx.Graph, k: int, features: np.ndarray = None, model_path: str = None
) -> Set[int]:
    """基于GAT模型的关键节点选择

    参数:
        graph: 网络图
        k: 选择的节点数量
        features: 节点特征矩阵，如果为None则使用默认特征
        model_path: 预训练模型路径
    """
    if features is None:
        # 如果没有提供特征，使用默认特征生成方法
        dataset = SNAPDataset("epinions", n_simulations=100)
        data = dataset.to_pyg_data()
        features = data.x.numpy()

    # 创建边索引
    edge_index = np.array(list(graph.edges())).T

    # 创建模型
    model = GATModel(
        in_channels=features.shape[1], hidden_channels=32, out_channels=1, heads=8
    )

    # 如果提供了模型路径，加载预训练模型
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        # 从checkpoint中提取模型状态字典
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # 转换为PyTorch张量
    x = torch.FloatTensor(features)
    edge_index = torch.LongTensor(edge_index)

    model.eval()
    with torch.no_grad():
        predictions = model(x, edge_index)
        scores = predictions.cpu().numpy().flatten()
        # 返回得分最高的k个节点
        return set(np.argsort(scores)[-k:].tolist())


def simulate_propagation(
    graph: nx.Graph, seed_nodes: Set[int], activation_prob: float = 0.1
) -> Dict:
    """模拟信息传播过程"""
    activation_times = {node: 0 for node in seed_nodes}  # 种子节点在时间0被激活
    activation_paths = {node: [node] for node in seed_nodes}
    current_time = 0
    newly_activated = seed_nodes

    while newly_activated:
        current_time += 1
        next_activated = set()

        for node in newly_activated:
            for neighbor in graph.neighbors(node):
                if neighbor not in activation_times:  # 如果邻居节点未被激活
                    if np.random.random() < activation_prob:  # 激活概率
                        activation_times[neighbor] = current_time
                        activation_paths[neighbor] = activation_paths[node] + [neighbor]
                        next_activated.add(neighbor)

        newly_activated = next_activated

    # 模拟多次传播结果
    propagation_results = []
    for _ in range(3):  # 进行3次传播模拟
        results = []
        for node in graph.nodes():
            if node in activation_times:
                results.append(activation_times[node] / current_time)
            else:
                results.append(0.0)
        propagation_results.append(results)

    return {
        "activation_times": activation_times,
        "activation_paths": activation_paths,
        "propagation_results": propagation_results,
    }


def k_shell_selection(graph: nx.Graph, k: int) -> Set[int]:
    """基于K-壳分解的关键节点选择"""
    # 计算每个节点的k-shell值
    k_shell = nx.core_number(graph)
    # 返回k-shell值最大的k个节点
    return {
        node
        for node, _ in sorted(k_shell.items(), key=lambda x: x[1], reverse=True)[:k]
    }


def closeness_selection(graph: nx.Graph, k: int) -> Set[int]:
    """基于接近中心性的关键节点选择（使用改进的Wasserman-Faust公式）"""
    # 使用改进的Wasserman-Faust公式，这个公式对于不连通图有更好的处理
    closeness = nx.closeness_centrality(
        graph, wf_improved=True  # 使用改进的Wasserman-Faust公式
    )
    return {
        node
        for node, _ in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:k]
    }


def eigenvector_selection(graph: nx.Graph, k: int) -> Set[int]:
    """基于特征向量中心性的关键节点选择"""
    eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
    return {
        node
        for node, _ in sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:k]
    }


def clustering_selection(graph: nx.Graph, k: int) -> Set[int]:
    """基于局部聚类系数的关键节点选择

    选择聚类系数最高的节点作为种子节点
    """
    clustering = nx.clustering(graph)
    return {
        node
        for node, _ in sorted(clustering.items(), key=lambda x: x[1], reverse=True)[:k]
    }


def structural_holes_selection(graph: nx.Graph, k: int) -> Set[int]:
    """基于结构洞指标的关键节点选择

    选择约束度最低的节点作为种子节点
    """

    def local_constraint(node):
        # 获取2-hop邻居
        neighbors = set(graph.neighbors(node))
        second_neighbors = set()
        for neighbor in neighbors:
            second_neighbors.update(graph.neighbors(neighbor))
        second_neighbors -= neighbors  # 移除一阶邻居

        # 计算局部约束度
        constraint = 0
        for neighbor in neighbors:
            # 计算与共同邻居的连接数
            common_neighbors = set(graph.neighbors(neighbor)) & neighbors
            if len(common_neighbors) > 0:
                constraint += (len(common_neighbors) / len(neighbors)) ** 2

        return constraint / len(neighbors) if neighbors else 0

    # 计算所有节点的局部约束度
    constraints = {node: local_constraint(node) for node in graph.nodes()}

    # 返回约束度最低的k个节点
    return {node for node, _ in sorted(constraints.items(), key=lambda x: x[1])[:k]}


def greedy_imm_selection(graph: nx.Graph, k: int, n_simulations: int = 5) -> Set[int]:
    """基于贪心算法的影响力最大化（使用CELF优化）

    使用CELF（Cost-Effective Lazy Forward）算法来加速贪心选择
    """

    def estimate_influence(seed_nodes: Set[int]) -> float:
        total_activated = 0
        for _ in range(n_simulations):
            active = set(seed_nodes)
            newly_active = set(seed_nodes)

            while newly_active:
                next_active = set()
                for node in newly_active:
                    for neighbor in graph.neighbors(node):
                        if neighbor not in active:
                            if np.random.random() < 0.1:  # 激活概率
                                next_active.add(neighbor)
                active.update(next_active)
                newly_active = next_active

            total_activated += len(active)

        return total_activated / n_simulations

    # CELF优化
    selected = set()
    remaining = set(graph.nodes())

    # 初始化边际增益
    marginal_gains = {}
    for node in remaining:
        marginal_gains[node] = estimate_influence({node})

    # 贪心选择
    for _ in range(k):
        if not remaining:
            break

        # 找到边际增益最大的节点
        best_node = max(remaining, key=lambda x: marginal_gains[x])

        # 更新其他节点的边际增益
        if len(selected) > 0:
            for node in remaining - {best_node}:
                # 使用CELF优化：如果当前边际增益小于已选节点的最小边际增益，则跳过
                if marginal_gains[node] <= marginal_gains[best_node]:
                    continue
                # 重新计算边际增益
                marginal_gains[node] = estimate_influence(
                    selected | {node}
                ) - estimate_influence(selected)

        selected.add(best_node)
        remaining.remove(best_node)
        del marginal_gains[best_node]

    return selected


def compare_algorithms(
    graph: nx.Graph,
    k: int = 10,
    features: np.ndarray = None,
    gat_model_path: str = None,
    n_rounds: int = 20,
) -> Dict:
    """比较不同算法的性能

    参数:
        graph: 网络图
        k: 选择的节点数量
        features: 节点特征矩阵，用于GAT模型
        gat_model_path: GAT模型路径
        n_rounds: 传播模拟轮次
    """
    # 测试不同算法
    algorithms = {
        "Degree Centrality": degree_centrality_selection,
        "PageRank": pagerank_selection,
        "Betweenness Centrality": betweenness_selection,
        "K-Shell": k_shell_selection,
        # "Closeness Centrality": closeness_selection,
        "Eigenvector Centrality": eigenvector_selection,
        "Clustering Coefficient": clustering_selection,
        "Structural Holes": structural_holes_selection,  # 使用近似算法
        # "Greedy IMM": lambda g, k: greedy_imm_selection(
        #     g, k, n_simulations=5
        # ),  # 使用CELF优化
    }

    # 如果提供了GAT模型路径，添加GAT算法
    if gat_model_path:
        if isinstance(gat_model_path, list):
            for model_path in gat_model_path:
                algorithms[f"GAT_{model_path.split('/')[-1].split('.')[0]}"] = (
                    lambda g, k: gat_selection(g, k, features, model_path)
                )
        else:
            algorithms["GAT"] = lambda g, k: gat_selection(
                g, k, features, gat_model_path
            )

    results = {}
    for name, algorithm in algorithms.items():
        print(f"\n测试 {name} 算法...")

        # 记录开始时间
        start_time = time.time()

        # 选择种子节点
        seed_nodes = algorithm(graph, k)

        # 记录计算时间
        computation_time = max(0.0, time.time() - start_time)  # 确保时间不为负

        # 创建模拟的模型结果
        model_results = {
            "activation_times": {node: 0 for node in seed_nodes},
            "activation_paths": {node: [node] for node in seed_nodes},
            "computation_time": computation_time,  # 添加计算时间到模型结果中
        }

        # 使用独立级联模型评估
        print("使用独立级联模型评估...")
        ic_metrics = evaluate_model_performance(
            graph=graph,
            model_results=model_results,
            seed_nodes=seed_nodes,
            n_rounds=n_rounds,
            propagation_model="ic",
        )

        # 使用线性阈值模型评估
        print("使用线性阈值模型评估...")
        lt_metrics = evaluate_model_performance(
            graph=graph,
            model_results=model_results,
            seed_nodes=seed_nodes,
            n_rounds=n_rounds,
            propagation_model="lt",
        )

        # 记录结果
        results[name] = {
            "seed_nodes": seed_nodes,
            "independent_cascade": ic_metrics,
            "linear_threshold": lt_metrics,
            "computation_time": computation_time,  # 使用同一个计算时间
        }

    return results


def print_comparison_results(results: Dict):
    """打印比较结果（Markdown格式）"""
    # 打印独立级联模型结果
    print("\n独立级联模型 (IC) 性能比较结果\n")

    # 定义表头
    headers = ["算法", "总激活节点数", "模型稳定性"]

    # 打印表头
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    # 打印数据行
    for name, result in results.items():
        metrics = result["independent_cascade"]
        print(
            f"| {name} | {metrics['total_activated']} | {metrics['model_stability']:.4f} |"
        )

    # 打印线性阈值模型结果
    print("\n线性阈值模型 (LT) 性能比较结果\n")

    # 打印表头
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    # 打印数据行
    for name, result in results.items():
        metrics = result["linear_threshold"]
        print(
            f"| {name} | {metrics['total_activated']} | {metrics['model_stability']:.4f} |"
        )

    # 打印种子节点结构特征
    print("\n种子节点结构特征\n")

    # 定义结构特征表头
    struct_headers = ["算法", "平均度", "平均聚类系数", "平均距离", "计算时间"]

    # 打印表头
    print("| " + " | ".join(struct_headers) + " |")
    print("| " + " | ".join(["---"] * len(struct_headers)) + " |")

    # 打印结构特征数据
    for name, result in results.items():
        metrics = result[
            "independent_cascade"
        ]  # 使用IC模型的指标，因为结构特征与传播模型无关
        degree_dist = metrics["degree_distribution"]
        clustering_dist = metrics["clustering_distribution"]

        avg_degree = sum(degree_dist.values()) / len(degree_dist)
        avg_clustering = sum(clustering_dist.values()) / len(clustering_dist)
        avg_distance = metrics["average_distance"]
        computation_time = metrics["computation_time"]

        print(
            f"| {name} | {avg_degree:.2f} | {avg_clustering:.4f} | {avg_distance:.2f} | {computation_time:.4f} |"
        )


def main():
    """主函数"""
    # 加载网络数据和特征
    print("加载网络数据...")
    graph, features = load_network("data/processed/epinions_processed.pkl")

    # 比较算法性能（包括GAT模型）
    print("开始比较算法性能...")
    results = compare_algorithms(
        graph,
        k=10,
        features=features,
        gat_model_path=[
            "outputs/twitter/model/twitter_gat_model.pt",
            "outputs/epinions/model/epinions_gat_model.pt",
            "outputs/facebook/model/facebook_gat_model.pt",
        ],
    )

    print(f"\n网络节点数: {graph.number_of_nodes()}")
    print(f"网络边数: {graph.number_of_edges()}")
    # 打印结果
    print_comparison_results(results)


if __name__ == "__main__":
    main()
