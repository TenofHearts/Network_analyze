import unittest
import networkx as nx
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from metrics import (
    PropagationMetrics,
    StabilityMetrics,
    StructuralMetrics,
)
from src.models.propagation import IndependentCascade, LinearThreshold


def calculate_structural_metrics(
    graph: nx.Graph,
    seed_nodes: set,
    communities: list = None,
) -> dict:
    """
    计算网络的结构特征指标

    参数:
        graph: 网络图
        seed_nodes: 种子节点集合
        communities: 社区列表，可选

    返回:
        包含结构特征指标的字典
    """
    struct_metrics = StructuralMetrics(graph)

    degree_dist = struct_metrics.calculate_degree_distribution(seed_nodes)
    clustering_dist = struct_metrics.calculate_clustering_distribution(seed_nodes)
    avg_distance = struct_metrics.calculate_average_distance(seed_nodes)

    # 计算社区覆盖（如果提供了社区信息）
    community_coverage = None
    if communities:
        community_coverage = struct_metrics.calculate_community_coverage(
            seed_nodes, communities
        )

    return {
        "degree_distribution": degree_dist,
        "clustering_distribution": clustering_dist,
        "average_distance": avg_distance,
        "community_coverage": community_coverage,
    }


def evaluate_model_performance(
    graph: nx.Graph,
    seed_nodes: set,
    n_rounds: int = 20,
    propagation_model: str = "ic",
) -> dict:
    """
    评估模型在给定网络上的表现

    参数:
        graph: 网络图
        seed_nodes: 种子节点集合
        n_rounds: 传播模拟轮次
        propagation_model: 传播模型类型，可选 "ic" 或 "lt"

    返回:
        包含所有评估指标的字典
    """
    # 初始化指标计算器
    prop_metrics = PropagationMetrics(graph)
    stab_metrics = StabilityMetrics()

    # 选择传播模型
    if propagation_model.lower() == "lt":
        propagation = LinearThreshold(graph)
        step = 10
    else:
        propagation = IndependentCascade(graph)
        step = 15

    # 进行多次传播模拟
    propagation_results = []
    all_activation_times = []
    all_activation_paths = []

    for _ in range(n_rounds):
        # 使用相同的种子节点进行传播
        result = propagation.simulate(list(seed_nodes))
        # 记录每轮传播的激活历史
        propagation_results.append(result["total_activated"])
        # 记录激活时间和路径
        all_activation_times.append(result.get("activation_times", {}))
        all_activation_paths.append(result.get("activation_paths", {}))

    # 计算传播效果指标
    # 使用所有轮次的平均激活节点
    total_activated = np.mean(np.array(propagation_results))

    # 计算稳定性指标
    stability = stab_metrics.calculate_model_stability(propagation_results)

    # 返回所有指标
    return {
        # 传播效果指标
        "total_activated": total_activated,
        # 稳定性指标
        "model_stability": stability,
    }
