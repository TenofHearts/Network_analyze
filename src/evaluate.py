import torch
import networkx as nx
import numpy as np
import time
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from models.gat import GATModel
from models.propagation import IndependentCascade
from data.dataset import SNAPDataset
from utils.metrics import calculate_spread_metrics


def degree_centrality_selection(graph: nx.Graph, k: int) -> List[int]:
    """基于度中心性选择节点"""
    degrees = dict(graph.degree())
    return sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:k]


def pagerank_selection(graph: nx.Graph, k: int) -> List[int]:
    """基于PageRank选择节点"""
    pagerank = nx.pagerank(graph)
    return sorted(pagerank.keys(), key=lambda x: pagerank[x], reverse=True)[:k]


def k_shell_selection(graph: nx.Graph, k: int) -> List[int]:
    """基于k-壳分解选择节点"""
    k_shell = nx.core_number(graph)
    return sorted(k_shell.keys(), key=lambda x: k_shell[x], reverse=True)[:k]


def greedy_imm_selection(
    graph: nx.Graph, k: int, n_simulations: int = 100
) -> List[int]:
    """贪心最大影响力算法"""
    selected_nodes = []
    remaining_nodes = set(graph.nodes())

    for _ in tqdm(range(k), desc="贪心选择节点"):
        best_node = None
        best_spread = -1

        # 对每个候选节点进行模拟
        for node in remaining_nodes:
            total_spread = 0
            for _ in range(n_simulations):
                ic = IndependentCascade(graph)
                result = ic.simulate([node] + selected_nodes)
                total_spread += result["total_activated"]

            avg_spread = total_spread / n_simulations
            if avg_spread > best_spread:
                best_spread = avg_spread
                best_node = node

        selected_nodes.append(best_node)
        remaining_nodes.remove(best_node)

    return selected_nodes


def gat_selection(model: GATModel, data: torch.Tensor, k: int) -> List[int]:
    """基于GAT模型选择节点"""
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index)
        scores = predictions.cpu().numpy().flatten()
        return np.argsort(scores)[-k:].tolist()


def evaluate_selection(
    graph: nx.Graph, selected_nodes: List[int], n_simulations: int = 100
) -> Dict[str, float]:
    """评估节点选择的效果"""
    propagation_results = []
    for _ in range(n_simulations):
        ic = IndependentCascade(graph)
        result = ic.simulate(selected_nodes)
        propagation_results.append(result)

    metrics = calculate_spread_metrics(propagation_results)
    return metrics


def run_evaluation(dataset_name: str, k: int, n_simulations: int = 100):
    """运行评估"""
    print(f"\n评估数据集: {dataset_name}")
    print(f"选择节点数: {k}")
    print(f"模拟次数: {n_simulations}")

    # 加载数据集
    dataset = SNAPDataset(dataset_name, n_simulations=n_simulations)
    graph = dataset.load_graph()
    data = dataset.to_pyg_data()

    # 评估结果存储
    results = []

    # 1. 度中心性
    print("\n评估度中心性...")
    start_time = time.time()
    degree_nodes = degree_centrality_selection(graph, k)
    degree_time = time.time() - start_time
    degree_metrics = evaluate_selection(graph, degree_nodes, n_simulations)
    results.append({"算法": "度中心性", "计算时间": degree_time, **degree_metrics})

    # 2. PageRank
    print("\n评估PageRank...")
    start_time = time.time()
    pagerank_nodes = pagerank_selection(graph, k)
    pagerank_time = time.time() - start_time
    pagerank_metrics = evaluate_selection(graph, pagerank_nodes, n_simulations)
    results.append({"算法": "PageRank", "计算时间": pagerank_time, **pagerank_metrics})

    # 3. k-壳分解
    print("\n评估k-壳分解...")
    start_time = time.time()
    kshell_nodes = k_shell_selection(graph, k)
    kshell_time = time.time() - start_time
    kshell_metrics = evaluate_selection(graph, kshell_nodes, n_simulations)
    results.append({"算法": "k-壳分解", "计算时间": kshell_time, **kshell_metrics})

    # 4. 贪心算法
    print("\n评估贪心算法...")
    start_time = time.time()
    greedy_nodes = greedy_imm_selection(
        graph, k, n_simulations=10
    )  # 减少模拟次数以加快速度
    greedy_time = time.time() - start_time
    greedy_metrics = evaluate_selection(graph, greedy_nodes, n_simulations)
    results.append({"算法": "贪心算法", "计算时间": greedy_time, **greedy_metrics})

    # 5. GAT模型
    print("\n评估GAT模型...")
    # 创建并加载模型
    model = GATModel(
        in_channels=data.x.size(1), hidden_channels=32, out_channels=1, heads=8
    )
    # 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    start_time = time.time()
    # 训练模型
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

    # 选择节点
    gat_nodes = gat_selection(model, data, k)
    gat_time = time.time() - start_time
    gat_metrics = evaluate_selection(graph, gat_nodes, n_simulations)
    results.append({"算法": "GAT", "计算时间": gat_time, **gat_metrics})

    # 转换为DataFrame并保存结果
    df = pd.DataFrame(results)
    print("\n评估结果：")
    print(df)

    # 保存结果
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(f"outputs/{dataset_name}_evaluation_results.csv", index=False)

    # 绘制对比图
    plt.figure(figsize=(15, 10))

    # 1. 平均激活节点数对比
    plt.subplot(2, 2, 1)
    plt.bar(df["算法"], df["mean_activated"])
    plt.title("平均激活节点数")
    plt.xticks(rotation=45)

    # 2. 计算时间对比
    plt.subplot(2, 2, 2)
    plt.bar(df["算法"], df["计算时间"])
    plt.title("计算时间 (秒)")
    plt.xticks(rotation=45)

    # 3. 传播步数对比
    plt.subplot(2, 2, 3)
    plt.bar(df["算法"], df["mean_steps"])
    plt.title("平均传播步数")
    plt.xticks(rotation=45)

    # 4. 激活节点数标准差对比
    plt.subplot(2, 2, 4)
    plt.bar(df["算法"], df["std_activated"])
    plt.title("激活节点数标准差")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"outputs/{dataset_name}_comparison.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="评估不同关键节点选择算法")
    parser.add_argument(
        "--dataset",
        type=str,
        default="facebook",
        choices=["facebook", "twitter", "epinions"],
        help="选择数据集",
    )
    parser.add_argument("--k", type=int, default=10, help="选择的关键节点数量")
    parser.add_argument("--n_simulations", type=int, default=100, help="传播模拟次数")
    args = parser.parse_args()

    run_evaluation(args.dataset, args.k, args.n_simulations)


if __name__ == "__main__":
    main()
