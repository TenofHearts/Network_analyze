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
        self, activation_times: Dict[int, int], max_steps: int = 5
    ) -> float:
        """计算传播速度：在固定步数内覆盖的节点数量比例

        参数:
            activation_times: 节点激活时间字典
            max_steps: 最大传播步数，默认为5步

        返回:
            在max_steps步内被激活的节点数量比例
        """
        # 计算在max_steps步内被激活的节点数量
        activated_in_steps = sum(
            1 for time in activation_times.values() if time <= max_steps
        )
        # 返回覆盖率
        return activated_in_steps / self.graph.number_of_nodes()

    def calculate_propagation_depth(
        self, activation_paths: Dict[int, List[int]]
    ) -> int:
        """计算传播深度：信息传播的最大跳数"""
        max_depth = 0
        for path in activation_paths.values():
            max_depth = max(max_depth, len(path) - 1)
        return max_depth


class StabilityMetrics:
    def __init__(self):
        pass

    def calculate_model_stability(self, results: List[List[float]]) -> float:
        """计算模型结果稳定性：多次模拟的标准差"""
        if not results or not results[0]:
            return 0.0

        # 计算每次模拟的平均激活率
        mean_activations = [np.mean(run) for run in results]

        # 计算变异系数（标准差/平均值）作为稳定性指标
        if np.mean(mean_activations) == 0:
            return 0.0

        cv = np.std(mean_activations) / np.mean(mean_activations)
        # 将变异系数转换为稳定性分数（0-1之间，1表示最稳定）
        stability = 1.0 / (1.0 + cv)
        return stability

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
        self.end_time = None

    def start_timer(self):
        """开始计时"""
        self.start_time = time.time()
        self.end_time = None

    def stop_timer(self):
        """停止计时"""
        self.end_time = time.time()

    def calculate_computation_time(self) -> float:
        """计算节点重要性计算时间"""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            self.stop_timer()
        return max(0.0, self.end_time - self.start_time)
