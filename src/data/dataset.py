import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Tuple, Optional, Dict, List
import urllib.request
import gzip
import shutil
from tqdm import tqdm
import pickle
from pathlib import Path

from models.propagation import IndependentCascade
from config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_URLS


class SNAPDataset:
    """Stanford SNAP数据集加载器"""

    def __init__(self, dataset_name: str, n_simulations: int = 100):
        """
        初始化数据集加载器

        参数:
            dataset_name: 数据集名称 ('facebook', 'twitter', 'epinions')
            n_simulations: IC模型模拟次数
        """
        self.dataset_name = dataset_name
        self.raw_file = RAW_DATA_DIR / f"{dataset_name}.txt"
        self.n_simulations = n_simulations
        self.cache_file = PROCESSED_DATA_DIR / f"{dataset_name}_processed.pkl"

        # 下载数据集（如果不存在）
        if not os.path.exists(self.raw_file):
            self._download_dataset()

    def _download_dataset(self):
        """下载并解压数据集"""
        if self.dataset_name not in DATASET_URLS:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")

        url = DATASET_URLS[self.dataset_name]
        gz_file = f"{self.raw_file}.gz"

        print(f"正在下载数据集: {self.dataset_name}")
        urllib.request.urlretrieve(url, gz_file)

        print("正在解压数据集...")
        with gzip.open(gz_file, "rb") as f_in:
            with open(self.raw_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(gz_file)
        print("数据集下载完成")

    def load_graph(self) -> nx.Graph:
        """加载NetworkX图对象"""
        print(f"正在加载图数据: {self.dataset_name}")
        if self.dataset_name == "epinions":
            # Epinions数据集有特殊的格式
            graph = nx.read_edgelist(
                self.raw_file, comments="#", create_using=nx.Graph()
            )
        else:
            graph = nx.read_edgelist(self.raw_file, create_using=nx.Graph())

        # 确保节点编号从0开始连续
        graph = nx.convert_node_labels_to_integers(graph, first_label=0)
        return graph

    def create_node_features(self, graph: nx.Graph) -> np.ndarray:
        """
        创建节点特征

        使用以下特征:
        1. 节点度
        2. 聚类系数
        3. PageRank值
        4. 二阶邻居数
        5. 节点度中心性
        6. 特征向量中心性的快速近似
        """
        print("正在生成节点特征...")
        n_nodes = len(graph)
        features = np.zeros((n_nodes, 6 if len(graph) < 5000 else 7))

        # 1. 节点度
        degrees = dict(graph.degree())
        features[:, 0] = [degrees[node] for node in range(n_nodes)]

        # 2. 聚类系数
        clustering = nx.clustering(graph)
        features[:, 1] = [clustering.get(node, 0) for node in range(n_nodes)]

        # 3. PageRank
        pagerank = nx.pagerank(graph, max_iter=100)  # 减少迭代次数以加快速度
        features[:, 2] = [pagerank[node] for node in range(n_nodes)]

        # 4. 二阶邻居数
        for node in range(n_nodes):
            neighbors = set(graph.neighbors(node))
            second_neighbors = set()
            for neighbor in neighbors:
                second_neighbors.update(graph.neighbors(neighbor))
            second_neighbors -= neighbors  # 移除一阶邻居
            features[node, 3] = len(second_neighbors)

        # 5. 节点度中心性（归一化的度）
        max_degree = max(degrees.values())
        features[:, 4] = [degrees[node] / max_degree for node in range(n_nodes)]

        # 6. 特征向量中心性的快速近似（使用幂迭代法，只迭代5次）
        def power_iteration(graph, n_iter=5):
            n = len(graph)
            x = np.ones(n) / n
            for _ in range(n_iter):
                x_new = np.zeros(n)
                for node in range(n):
                    for neighbor in graph.neighbors(node):
                        x_new[node] += x[neighbor]
                x = x_new / np.sum(x_new)
            return x

        eigenvector_approx = power_iteration(graph)
        features[:, 5] = eigenvector_approx

        # 7. 介数中心性(若节点数小于5,000)
        if len(graph) < 5000:
            betweenness = nx.betweenness_centrality(graph)
            features[:, 6] = [betweenness[node] for node in range(n_nodes)]
        else:
            features[:, 6] = 0

        # 标准化特征
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        return features

    def generate_ic_labels(self, graph: nx.Graph) -> np.ndarray:
        """
        通过IC模型多次模拟生成节点重要性标签

        参数:
            graph: NetworkX图对象

        返回:
            节点重要性标签
        """
        print(f"正在进行{self.n_simulations}次IC模拟生成标签...")
        n_nodes = len(graph)
        influence_scores = np.zeros(n_nodes)

        # 创建IC模型
        ic_model = IndependentCascade(graph)

        # 对每个节点进行多次模拟
        for node in tqdm(range(n_nodes)):
            total_activated = 0
            for _ in range(self.n_simulations):
                result = ic_model.simulate([node])
                total_activated += result["total_activated"]
            influence_scores[node] = total_activated / self.n_simulations

        # 标准化影响力分数
        influence_scores = (influence_scores - influence_scores.min()) / (
            influence_scores.max() - influence_scores.min() + 1e-8
        )
        return influence_scores

    def _load_processed_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """加载处理后的数据"""
        if os.path.exists(self.cache_file):
            print(f"从{self.cache_file}加载处理后的数据...")
            try:
                with open(self.cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    print("从缓存加载处理后的数据...")
                    return cached_data["features"], cached_data["labels"]
            except Exception as e:
                print(f"加载缓存数据失败: {e}")
        return None

    def _save_processed_data(self, features: np.ndarray, labels: np.ndarray):
        """保存处理后的数据"""
        cached_data = {
            "features": features,
            "labels": labels,
            "n_simulations": self.n_simulations,
            "dataset_name": self.dataset_name,
        }
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(cached_data, f)
            print(f"处理后的数据已保存到: {self.cache_file}")
        except Exception as e:
            print(f"保存缓存数据失败: {e}")

    def to_pyg_data(self) -> Data:
        """
        将图数据转换为PyTorch Geometric格式

        返回:
            PyG Data对象
        """
        # 尝试从缓存加载处理后的数据
        cached_data = self._load_processed_data()

        if cached_data is not None:
            features, labels = cached_data
        else:
            # 如果没有缓存，重新计算
            graph = self.load_graph()
            features = self.create_node_features(graph)
            labels = self.generate_ic_labels(graph)
            # 保存处理后的数据
            self._save_processed_data(features, labels)

        # 加载图数据（这个操作比较快，不需要缓存）
        graph = self.load_graph()

        # 创建边索引
        edge_index = np.array(list(graph.edges())).T

        # 转换为PyG Data对象
        data = Data(
            x=torch.FloatTensor(features),
            edge_index=torch.LongTensor(edge_index),
            y=torch.FloatTensor(labels).view(-1, 1),
        )

        return data
