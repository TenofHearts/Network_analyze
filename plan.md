# 项目策划书：基于GAT的社交网络关键节点预测与传播模拟

---

## 一、项目背景与意义

在社交网络中，一些节点在信息传播中起着关键作用。这些节点被称为"关键节点"。此类节点预测是社群模型、疫患控制、营销传播等领域的核心问题。本项目将利用图注意力网络(GAT)，对社交网络中节点的影响力进行预测，并进一步通过传播模型模拟，与传统的关键节点选择算法进行效果对比，以验证GAT在社会网络分析中的实际效能。

---

## 二、研究内容与方法

1. **数据集选择**：

   -  选用Stanford SNAP库中的社交网络数据，如Facebook、Epinions、Twitter社交图等。
   -  实现自动下载和解压功能，支持数据集的本地缓存。

2. **节点特征构造**：

   -  实现基于结构的手工特征生成，包括：
      - 节点度
      - 聚类系数
      - PageRank值
      - 二阶邻居数
      - 节点度中心性
      - 特征向量中心性的快速近似
   -  所有特征进行标准化处理，确保数值稳定性。

3. **基于GAT的关键节点预测**：

   - 构建图注意力网络模型，输入为节点的嵌入表示及邻居结构信息。
   - 模型通过多头注意力机制学习邻居节点对当前节点的重要性，并通过叠加注意力层捕捉多阶结构依赖。
   - 输出为每个节点的影响力分数，代表其作为传播种子节点的潜力。
   - 损失函数设计方面，采用半监督方式生成伪标签：
      - 通过独立级联（Independent Cascade, IC）模型进行多次传播模拟
      - 对每个节点作为单一传播源进行多次模拟
      - 统计每个节点作为种子时平均激活的节点数作为影响力分值
     - 对影响力分数进行归一化处理
   - 实现数据缓存机制，避免重复计算特征和标签。

4. **传播模拟对比**：

   - 采用Independent Cascade(IC)或Linear Threshold(LT)传播模型，模拟信息从选出的种子节点向整个社会网络的传播效果。
   - 将GAT给出的Top-k节点与经典方法(尤其是Degree、PageRank、Greedy IMM)的选择结果进行效果对比。

5. **算法评估指标**：

   - 传播效果指标：
     - 最终被激活节点数（覆盖率）
     - 传播速度（达到特定覆盖率所需时间）
     - 传播深度（信息传播的最大跳数）
     - 传播效率（单位时间内激活的节点数）
   
   - 稳定性指标：
     - 模型结果稳定性（多次模拟标准差）
     - 种子节点选择的一致性（不同参数设置下的重叠率）
     - 传播路径的多样性（不同传播路径的数量）
   
   - 结构特征指标：
     - 种子节点的度分布
     - 种子节点的聚类系数分布
     - 种子节点间的平均距离
     - 种子节点覆盖的社区数量
   
   - 计算效率指标：
     - 节点重要性计算时间
   
   <!-- - 对比分析指标：
     - 与基准方法的性能提升百分比
     - 不同规模网络下的可扩展性
     - 不同传播概率下的鲁棒性
     - 不同种子节点数量下的边际效益 -->

## 实验结果

网络节点数: 75879
网络边数: 405740

### 独立级联模型 (IC) 性能比较结果

$k=10$

| 算法 | 总激活节点数 | 模型稳定性 |
| --- | --- | --- |
| Degree Centrality | 22831.05 | 0.9948 |
| PageRank | 22846.85 | 0.9959 |
| Betweenness Centrality | 22825.15 | 0.9964 |
| K-Shell | 22804.1 | 0.9951 |
| Eigenvector Centrality | 22840.55 | 0.9960 |
| Clustering Coefficient | 21713.25 | 0.8134 |
| Structural Holes | 21711.55 | 0.8134 |
| GAT_twitter_gat_model | 22846.4 | 0.9948 |
| GAT_epinions_gat_model | 22877.4 | 0.9942 |
| GAT_facebook_gat_model | 22870.35 | 0.9960 |

$k = 5$

| 算法 | 总激活节点数 | 模型稳定性 |
| --- | --- | --- |
| Degree Centrality | 22865.0 | 0.9950 |
| PageRank | 22924.7 | 0.9961 |
| Betweenness Centrality | 22853.0 | 0.9958 |
| K-Shell | 22891.25 | 0.9965 |
| Eigenvector Centrality | 22898.5 | 0.9970 |
| Clustering Coefficient | 19420.75 | 0.7042 |
| Structural Holes | 17097.2 | 0.6340 |
| GAT_twitter_gat_model | 22905.7 | 0.9962 |
| GAT_epinions_gat_model | 22899.2 | 0.9964 |
| GAT_facebook_gat_model | 22874.6 | 0.9964 |

$k = 1$

| 算法 | 总激活节点数 | 模型稳定性 |
| --- | --- | --- |
| Degree Centrality | 22847.65 | 0.9955 |
| PageRank | 22898.8 | 0.9961 |
| Betweenness Centrality | 22870.9 | 0.9957 |
| K-Shell | 22814.0 | 0.9952 |
| Eigenvector Centrality | 22873.9 | 0.9968 |
| Clustering Coefficient | 2277.65 | 0.2501 |
| Structural Holes | 10257.25 | 0.4750 |
| GAT_twitter_gat_model | 22819.9 | 0.9953 |
| GAT_epinions_gat_model | 22879.5 | 0.9959 |
| GAT_facebook_gat_model | 22881.65 | 0.9964 |

### 线性阈值模型 (LT) 性能比较结果

$k = 10$

| 算法 | 总激活节点数 | 模型稳定性 |
| --- | --- | --- |
| Degree Centrality | 3844.45 | 0.9976 |
| PageRank | 3803.3 | 0.9964 |
| Betweenness Centrality | 3838.0 | 0.9970 |
| K-Shell | 3647.95 | 0.9959 |
| Eigenvector Centrality | 3834.7 | 0.9950 |
| Clustering Coefficient | 10.0 | 1.0000 |
| Structural Holes | 10.0 | 1.0000 |
| GAT_twitter_gat_model | 3770.8 | 0.9965 |
| GAT_epinions_gat_model | 3768.4 | 0.9967 |
| GAT_facebook_gat_model | 3774.45 | 0.9960 |

$k = 5$

| 算法 | 总激活节点数 | 模型稳定性 |
| --- | --- | --- |
| Degree Centrality | 3746.7 | 0.9972 |
| PageRank | 3732.95 | 0.9967 |
| Betweenness Centrality | 3727.8 | 0.9968 |
| K-Shell | 3337.55 | 0.9824 |
| Eigenvector Centrality | 3766.95 | 0.9967 |
| Clustering Coefficient | 5.0 | 1.0000 |
| Structural Holes | 5.0 | 1.0000 |
| GAT_twitter_gat_model | 3574.8 | 0.9927 |
| GAT_epinions_gat_model | 3593.0 | 0.9933 |
| GAT_facebook_gat_model | 3569.05 | 0.9929 |

$k = 1$

| 算法 | 总激活节点数 | 模型稳定性 |
| --- | --- | --- |
| Degree Centrality | 1.0 | 1.0000 |
| PageRank | 1.0 | 1.0000 |
| Betweenness Centrality | 1.0 | 1.0000 |
| K-Shell | 1.0 | 1.0000 |
| Eigenvector Centrality | 1.0 | 1.0000 |
| Clustering Coefficient | 1.0 | 1.0000 |
| Structural Holes | 1.0 | 1.0000 |
| GAT_twitter_gat_model | 1.0 | 1.0000 |
| GAT_epinions_gat_model | 1.0 | 1.0000 |
| GAT_facebook_gat_model | 1.0 | 1.0000 |

### 种子节点结构特征

$k = 10$

| 算法 | 平均度 | 平均聚类系数 | 平均距离 | 计算时间 |
| --- | --- | --- | --- | --- |
| Degree Centrality | 1608.70 | 0.0211 | 1.27 | 0.0423 |
| PageRank | 1407.30 | 0.0136 | 1.64 | 1.0891 |
| Betweenness Centrality | 1553.70 | 0.0199 | 1.29 | 35.5783 |
| K-Shell | 600.30 | 0.1290 | 1.38 | 0.7075 |
| Eigenvector Centrality | 1451.80 | 0.0350 | 1.20 | 1.4822 |
| Clustering Coefficient | 2.40 | 1.0000 | 3.31 | 5.0315 |
| Structural Holes | 2.10 | 0.0000 | 3.00 | 8.3645 |
| GAT_twitter_gat_model | 1228.50 | 0.0207 | 1.60 | 2.2576 |
| GAT_epinions_gat_model | 1228.50 | 0.0207 | 1.60 | 1.7004 |
| GAT_facebook_gat_model | 1228.50 | 0.0207 | 1.60 | 1.6282 |

$k = 5$

| 算法 | 平均度 | 平均聚类系数 | 平均距离 | 计算时间 |
| --- | --- | --- | --- | --- |
| Degree Centrality | 1958.60 | 0.0164 | 1.20 | 0.0378 |
| PageRank | 1831.20 | 0.0141 | 1.50 | 0.9555 |
| Betweenness Centrality | 1742.40 | 0.0161 | 1.30 | 19.6071 |
| K-Shell | 311.20 | 0.1757 | 1.10 | 0.6924 |
| Eigenvector Centrality | 1751.40 | 0.0282 | 1.00 | 1.4705 |
| Clustering Coefficient | 2.80 | 1.0000 | 3.00 | 7.9775 |
| Structural Holes | 2.20 | 0.0000 | 3.00 | 5.4000 |
| GAT_twitter_gat_model | 1294.00 | 0.0186 | 1.70 | 2.1426 |
| GAT_epinions_gat_model | 1294.00 | 0.0186 | 1.70 | 1.6378 |
| GAT_facebook_gat_model | 1294.00 | 0.0186 | 1.70 | 1.7189 |

$k = 1$

| 算法 | 平均度 | 平均聚类系数 | 平均距离 | 计算时间 |
| --- | --- | --- | --- | --- |
| Degree Centrality | 3044.00 | 0.0102 | 0.00 | 0.0380 |
| PageRank | 3044.00 | 0.0102 | 0.00 | 0.9750 |
| Betweenness Centrality | 1626.00 | 0.0191 | 0.00 | 4.0680 |
| K-Shell | 682.00 | 0.0537 | 0.00 | 0.8645 |
| Eigenvector Centrality | 3044.00 | 0.0102 | 0.00 | 1.4824 |
| Clustering Coefficient | 3.00 | 1.0000 | 0.00 | 8.2485 |
| Structural Holes | 4.00 | 0.0000 | 0.00 | 4.0779 |
| GAT_twitter_gat_model | 1682.00 | 0.0077 | 0.00 | 2.2342 |
| GAT_epinions_gat_model | 1682.00 | 0.0077 | 0.00 | 1.6342 |
| GAT_facebook_gat_model | 1682.00 | 0.0077 | 0.00 | 1.6646 |