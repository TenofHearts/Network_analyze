import unittest
import networkx as nx
import numpy as np
from metrics import (
    PropagationMetrics,
    StabilityMetrics,
    StructuralMetrics,
    EfficiencyMetrics,
)


def evaluate_model_performance(
    graph: nx.Graph, model_results: dict, seed_nodes: set, communities: list = None
) -> dict:
    """
    评估模型在给定网络上的表现

    参数:
        graph: 网络图
        model_results: 模型输出结果，包含以下字段：
            - activation_times: Dict[int, int] 节点激活时间
            - activation_paths: Dict[int, List[int]] 传播路径
            - propagation_results: List[List[float]] 多次传播结果
        seed_nodes: 种子节点集合
        communities: 社区列表，可选

    返回:
        包含所有评估指标的字典
    """
    # 初始化指标计算器
    prop_metrics = PropagationMetrics(graph)
    stab_metrics = StabilityMetrics()
    struct_metrics = StructuralMetrics(graph)
    eff_metrics = EfficiencyMetrics()

    # 开始计时
    eff_metrics.start_timer()

    # 计算传播效果指标
    activated_nodes = set(model_results["activation_times"].keys())
    coverage = prop_metrics.calculate_coverage(activated_nodes)
    speed = prop_metrics.calculate_propagation_speed(
        model_results["activation_times"], target_coverage=0.5
    )
    depth = prop_metrics.calculate_propagation_depth(model_results["activation_paths"])
    efficiency = prop_metrics.calculate_propagation_efficiency(
        activated_nodes, max(model_results["activation_times"].values())
    )

    # 计算稳定性指标
    stability = stab_metrics.calculate_model_stability(
        model_results["propagation_results"]
    )

    # 计算结构特征指标
    degree_dist = struct_metrics.calculate_degree_distribution(seed_nodes)
    clustering_dist = struct_metrics.calculate_clustering_distribution(seed_nodes)
    avg_distance = struct_metrics.calculate_average_distance(seed_nodes)

    # 计算社区覆盖（如果提供了社区信息）
    community_coverage = None
    if communities:
        community_coverage = struct_metrics.calculate_community_coverage(
            seed_nodes, communities
        )

    # 计算计算时间
    computation_time = eff_metrics.calculate_computation_time()

    # 返回所有指标
    return {
        # 传播效果指标
        "coverage": coverage,
        "propagation_speed": speed,
        "propagation_depth": depth,
        "propagation_efficiency": efficiency,
        # 稳定性指标
        "model_stability": stability,
        # 结构特征指标
        "degree_distribution": degree_dist,
        "clustering_distribution": clustering_dist,
        "average_distance": avg_distance,
        "community_coverage": community_coverage,
        # 计算效率指标
        "computation_time": computation_time,
    }
