import numpy as np
from typing import List, Dict, Set, Tuple
import networkx as nx
from collections import defaultdict
import time


class PropagationMetrics:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def calculate_coverage(self, activated_nodes: Set[int]) -> float:
        """计算覆盖率：最终被激活节点数占总节点数的比例"""
        return len(activated_nodes) / self.graph.number_of_nodes()

    def calculate_propagation_speed(
        self, activation_times: Dict[int, int], target_coverage: float
    ) -> int:
        """计算传播速度：达到特定覆盖率所需的时间步数"""
        total_nodes = self.graph.number_of_nodes()
        target_nodes = int(total_nodes * target_coverage)

        # 按激活时间排序
        sorted_times = sorted(activation_times.values())
        if len(sorted_times) >= target_nodes:
            return sorted_times[target_nodes - 1]
        return float("inf")

    def calculate_propagation_depth(
        self, activation_paths: Dict[int, List[int]]
    ) -> int:
        """计算传播深度：信息传播的最大跳数"""
        max_depth = 0
        for path in activation_paths.values():
            max_depth = max(max_depth, len(path) - 1)
        return max_depth

    def calculate_propagation_efficiency(
        self, activated_nodes: Set[int], total_time: int
    ) -> float:
        """计算传播效率：单位时间内激活的节点数"""
        return len(activated_nodes) / total_time if total_time > 0 else 0


class StabilityMetrics:
    def __init__(self):
        pass

    def calculate_model_stability(self, results: List[List[float]]) -> float:
        """计算模型结果稳定性：多次模拟的标准差"""
        return np.std([np.mean(run) for run in results])

    def calculate_seed_consistency(self, seed_sets: List[Set[int]]) -> float:
        """计算种子节点选择的一致性：不同参数设置下的重叠率"""
        if not seed_sets:
            return 0.0

        total_overlap = 0
        total_comparisons = 0

        for i in range(len(seed_sets)):
            for j in range(i + 1, len(seed_sets)):
                overlap = len(seed_sets[i] & seed_sets[j])
                union = len(seed_sets[i] | seed_sets[j])
                if union > 0:
                    total_overlap += overlap / union
                    total_comparisons += 1

        return total_overlap / total_comparisons if total_comparisons > 0 else 0.0

    def calculate_path_diversity(self, all_paths: List[List[List[int]]]) -> float:
        """计算传播路径的多样性：不同传播路径的数量"""
        unique_paths = set()
        for paths in all_paths:
            for path in paths:
                unique_paths.add(tuple(path))
        return len(unique_paths)


class StructuralMetrics:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def calculate_degree_distribution(self, seed_nodes: Set[int]) -> Dict[int, int]:
        """计算种子节点的度分布"""
        return {node: self.graph.degree(node) for node in seed_nodes}

    def calculate_clustering_distribution(
        self, seed_nodes: Set[int]
    ) -> Dict[int, float]:
        """计算种子节点的聚类系数分布"""
        return {node: nx.clustering(self.graph, node) for node in seed_nodes}

    def calculate_average_distance(self, seed_nodes: Set[int]) -> float:
        """计算种子节点间的平均距离"""
        if len(seed_nodes) < 2:
            return 0.0

        total_distance = 0
        count = 0

        for i in seed_nodes:
            for j in seed_nodes:
                if i < j:  # 避免重复计算
                    try:
                        distance = nx.shortest_path_length(self.graph, i, j)
                        total_distance += distance
                        count += 1
                    except nx.NetworkXNoPath:
                        continue

        return total_distance / count if count > 0 else float("inf")

    def calculate_community_coverage(
        self, seed_nodes: Set[int], communities: List[Set[int]]
    ) -> int:
        """计算种子节点覆盖的社区数量"""
        covered_communities = set()
        for community in communities:
            if any(node in seed_nodes for node in community):
                covered_communities.add(tuple(sorted(community)))
        return len(covered_communities)


class EfficiencyMetrics:
    def __init__(self):
        self.start_time = None

    def start_timer(self):
        """开始计时"""
        self.start_time = time.time()

    def calculate_computation_time(self) -> float:
        """计算节点重要性计算时间"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
