import torch
import networkx as nx
import numpy as np
from typing import List, Set, Dict


class IndependentCascade:
    def __init__(self, graph: nx.Graph, activation_prob: float = 0.1):
        """
        独立级联传播模型

        参数:
            graph: NetworkX图对象
            activation_prob: 激活概率
        """
        self.graph = graph
        self.activation_prob = activation_prob

    def simulate(self, seed_nodes: List[int], max_steps: int = 10) -> Dict[str, float]:
        """
        模拟信息传播过程

        参数:
            seed_nodes: 种子节点列表
            max_steps: 最大传播步数

        返回:
            包含传播结果的字典
        """
        # 初始化激活节点集合
        active_nodes = set(seed_nodes)
        newly_active = set(seed_nodes)

        # 记录激活时间和路径
        activation_times = {node: 0 for node in seed_nodes}  # 种子节点在时间0被激活
        activation_paths = {node: [node] for node in seed_nodes}

        # 记录每个时间步的激活节点数
        activation_history = [len(active_nodes)]

        # 传播过程
        for step in range(max_steps):
            if not newly_active:
                break

            # 当前时间步可能被激活的节点
            candidates = set()
            for node in newly_active:
                candidates.update(self.graph.neighbors(node))
            candidates -= active_nodes

            # 尝试激活候选节点
            newly_active = set()
            for node in candidates:
                # 获取所有已激活的邻居节点
                active_neighbors = [
                    n for n in self.graph.neighbors(node) if n in active_nodes
                ]
                # 计算激活概率
                activation_prob = 1 - (1 - self.activation_prob) ** len(
                    active_neighbors
                )
                # 随机决定是否激活
                if np.random.random() < activation_prob:
                    newly_active.add(node)
                    # 记录激活时间和路径
                    activation_times[node] = step + 1
                    # 找到激活该节点的邻居（随机选择一个）
                    activator = np.random.choice(active_neighbors)
                    activation_paths[node] = activation_paths[activator] + [node]

            # 更新激活节点集合
            active_nodes.update(newly_active)
            activation_history.append(len(active_nodes))

        return {
            "total_activated": len(active_nodes),
            "activation_history": activation_history,
            "propagation_steps": len(activation_history) - 1,
            "activation_times": activation_times,
            "activation_paths": activation_paths,
        }


class LinearThreshold:
    def __init__(self, graph: nx.Graph, threshold_range: tuple = (0.7, 0.9)):
        """
        改进的线性阈值传播模型（优化版本）

        参数:
            graph: NetworkX图对象
            threshold_range: 节点阈值的范围，用于随机生成每个节点的阈值
        """
        self.graph = graph
        self.threshold_range = threshold_range

        # 预计算节点的度
        self.degrees = dict(graph.degree())
        max_degree = max(self.degrees.values())

        # 预计算节点的PageRank值
        self.pagerank = nx.pagerank(graph)

        # 预计算节点的聚类系数
        self.clustering = nx.clustering(graph)

        # 预计算边权重
        self.edge_weights = {}
        for u, v in graph.edges():
            # 简化的权重计算
            weight = 0.4 * (
                self.pagerank[u] + self.pagerank[v]
            ) / 2 + 0.6 * (  # PageRank因子
                self.degrees[u] + self.degrees[v]
            ) / (
                2 * max_degree
            )  # 度因子
            self.edge_weights[(u, v)] = weight
            self.edge_weights[(v, u)] = weight  # 无向图，双向权重相同

        # 为每个节点生成阈值
        self.thresholds = {}
        for node in graph.nodes():
            # 基础阈值
            base_threshold = np.random.uniform(*threshold_range)
            # 根据聚类系数调整阈值
            clustering_factor = self.clustering.get(node, 0)
            self.thresholds[node] = base_threshold + (clustering_factor * 0.2)

    def simulate(self, seed_nodes: List[int], max_steps: int = 10) -> Dict[str, float]:
        """
        模拟信息传播过程

        参数:
            seed_nodes: 种子节点列表
            max_steps: 最大传播步数

        返回:
            包含传播结果的字典
        """
        # 初始化激活节点集合
        active_nodes = set(seed_nodes)
        newly_active = set(seed_nodes)

        # 记录激活时间和路径
        activation_times = {node: 0 for node in seed_nodes}
        activation_paths = {node: [node] for node in seed_nodes}

        # 记录每个时间步的激活节点数
        activation_history = [len(active_nodes)]

        # 传播过程
        for step in range(max_steps):
            if not newly_active:
                break

            # 当前时间步可能被激活的节点
            candidates = set()
            for node in newly_active:
                candidates.update(self.graph.neighbors(node))
            candidates -= active_nodes

            # 尝试激活候选节点
            newly_active = set()
            for node in candidates:
                # 获取已激活邻居
                active_neighbors = [
                    n for n in self.graph.neighbors(node) if n in active_nodes
                ]
                if not active_neighbors:
                    continue

                # 计算总影响（使用预计算的边权重）
                total_influence = sum(
                    self.edge_weights.get((neighbor, node), 0) / (1.0 + 0.1 * step)
                    for neighbor in active_neighbors
                )

                # 添加随机扰动
                noise = np.random.normal(0, 0.1)  # 均值为0，标准差为0.1的高斯噪声
                total_influence = max(0, total_influence + noise)  # 确保影响不为负

                # 如果总影响超过阈值，则激活节点
                if total_influence >= self.thresholds[node]:
                    newly_active.add(node)
                    activation_times[node] = step + 1
                    # 找到影响最大的邻居作为激活者
                    activator = max(
                        active_neighbors,
                        key=lambda n: self.edge_weights.get((n, node), 0),
                    )
                    activation_paths[node] = activation_paths[activator] + [node]

            # 更新激活节点集合
            active_nodes.update(newly_active)
            activation_history.append(len(active_nodes))

        return {
            "total_activated": len(active_nodes),
            "activation_history": activation_history,
            "propagation_steps": len(activation_history) - 1,
            "activation_times": activation_times,
            "activation_paths": activation_paths,
        }
