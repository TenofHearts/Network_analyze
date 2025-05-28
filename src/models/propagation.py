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
                active_neighbors = [n for n in self.graph.neighbors(node) if n in active_nodes]
                # 计算激活概率
                activation_prob = 1 - (1 - self.activation_prob) ** len(active_neighbors)
                # 随机决定是否激活
                if np.random.random() < activation_prob:
                    newly_active.add(node)
            
            # 更新激活节点集合
            active_nodes.update(newly_active)
            activation_history.append(len(active_nodes))
        
        return {
            'total_activated': len(active_nodes),
            'activation_history': activation_history,
            'propagation_steps': len(activation_history) - 1
        } 