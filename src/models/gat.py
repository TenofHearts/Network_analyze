import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn import global_mean_pool


class GATModel(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        heads=8,
        dropout=0.6,
        use_residual=True,
        use_layer_norm=True,
        use_gatv2=True,
    ):
        """
        优化的图注意力网络模型，适用于大型网络

        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出维度
            heads: 注意力头数
            dropout: Dropout比率
            use_residual: 是否使用残差连接
            use_layer_norm: 是否使用层归一化
            use_gatv2: 是否使用GATv2（动态注意力）
        """
        super(GATModel, self).__init__()
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.use_gatv2 = use_gatv2

        # 选择GAT版本
        GATLayer = GATv2Conv if use_gatv2 else GATConv

        # 第一层GAT
        self.conv1 = GATLayer(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
            add_self_loops=False,  # 手动添加自环
        )

        # 第二层GAT
        self.conv2 = GATLayer(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            dropout=dropout,
            concat=False,
            add_self_loops=False,
        )

        # 输出层
        self.out_layer = nn.Linear(hidden_channels, out_channels)

        # 层归一化
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(hidden_channels * heads)
            self.norm2 = nn.LayerNorm(hidden_channels)

        # 残差连接的投影层
        if use_residual:
            self.residual1 = nn.Linear(in_channels, hidden_channels * heads)
            self.residual2 = nn.Linear(hidden_channels * heads, hidden_channels)

    def forward(self, x, edge_index, batch=None):
        """
        前向传播

        参数:
            x: 节点特征矩阵
            edge_index: 边索引矩阵
            batch: 批处理索引（用于图采样）
        """
        # 添加自环
        edge_index, _ = add_self_loops(edge_index)

        # 第一层GAT
        identity = x
        x = self.conv1(x, edge_index)

        if self.use_layer_norm:
            x = self.norm1(x)

        if self.use_residual:
            x = x + self.residual1(identity)

        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层GAT
        identity = x
        x = self.conv2(x, edge_index)

        if self.use_layer_norm:
            x = self.norm2(x)

        if self.use_residual:
            x = x + self.residual2(identity)

        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 输出层
        x = self.out_layer(x)

        # 如果提供了batch信息，进行图池化
        if batch is not None:
            x = global_mean_pool(x, batch)

        return x
