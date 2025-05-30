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
import multiprocessing as mp
from functools import partial
from itertools import chain

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

    def _create_process_safe_ic_model(self, graph: nx.Graph) -> IndependentCascade:
        """
        创建进程安全的IC模型实例

        参数:
            graph: NetworkX图对象

        返回:
            IC模型实例
        """
        # 创建图的深拷贝，确保每个进程有自己的图副本
        graph_copy = graph.copy()
        return IndependentCascade(graph_copy)

    def _process_node_batch(
        self, node_batch: List[int], graph: nx.Graph, process_id: int
    ) -> List[float]:
        """
        处理一批节点的影响力计算

        参数:
            node_batch: 要处理的节点列表
            graph: NetworkX图对象
            process_id: 进程ID

        返回:
            节点影响力分数列表
        """
        # 为每个进程创建独立的IC模型实例
        ic_model = self._create_process_safe_ic_model(graph)
        results = []

        # 使用进程ID创建独立的进度条
        pbar = tqdm(
            node_batch,
            desc=f"进程 {process_id} 计算节点影响力",
            position=process_id,
            leave=True,
        )

        for node in pbar:
            total_activated = 0
            for _ in range(self.n_simulations):
                result = ic_model.simulate([node])
                total_activated += result["total_activated"]
            results.append(total_activated / self.n_simulations)

        return results

    def generate_ic_labels(self, graph: nx.Graph) -> np.ndarray:
        """
        通过IC模型多次模拟生成节点重要性标签（并行版本，固定节点分配）

        参数:
            graph: NetworkX图对象

        返回:
            节点重要性标签
        """
        print(f"正在进行{self.n_simulations}次IC模拟生成标签（并行计算）...")
        n_nodes = len(graph)

        # 创建进程池
        n_cores = max(1, mp.cpu_count() - 4)  # 保留一个核心给系统
        print(f"使用 {n_cores} 个CPU核心进行并行计算")

        # 将节点按进程数分组
        node_batches = [[] for _ in range(n_cores)]
        for node in range(n_nodes):
            process_id = node % n_cores
            node_batches[process_id].append(node)

        # 使用进程池并行计算
        with mp.Pool(processes=n_cores) as pool:
            # 为每个进程分配ID并执行计算
            results = list(
                pool.starmap(
                    self._process_node_batch,
                    [(batch, graph, i) for i, batch in enumerate(node_batches)],
                )
            )

        # 将结果重新排序
        influence_scores = np.zeros(n_nodes)
        for batch_idx, batch_results in enumerate(results):
            for node_idx, score in enumerate(batch_results):
                original_node = node_batches[batch_idx][node_idx]
                influence_scores[original_node] = score

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
