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
        attn_dropout=0.2,
        use_skip_connection=True,
        activation="elu",
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
            attn_dropout: 注意力机制的dropout比率
            use_skip_connection: 是否使用跳跃连接
            activation: 激活函数类型 ('elu', 'relu', 'leaky_relu')
        """
        super(GATModel, self).__init__()
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.use_gatv2 = use_gatv2
        self.use_skip_connection = use_skip_connection
        self.activation_type = activation
        self.hidden_channels = hidden_channels
        self.heads = heads

        # 选择GAT版本
        GATLayer = GATv2Conv if use_gatv2 else GATConv

        # 第一层GAT
        self.conv1 = GATLayer(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=attn_dropout,
            concat=True,
            add_self_loops=False,
        )

        # 第二层GAT
        self.conv2 = GATLayer(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            dropout=attn_dropout,
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

        # 跳跃连接
        if use_skip_connection:
            self.skip_connection = nn.Linear(hidden_channels * heads, hidden_channels)

        # 激活函数
        if activation == "elu":
            self.activation = F.elu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = lambda x: F.leaky_relu(x, negative_slope=0.2)
        else:
            raise ValueError(f"不支持的激活函数类型: {activation}")

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

        # 保存输入用于跳跃连接
        identity = x

        # 第一层GAT
        x = self.conv1(x, edge_index)

        if self.use_layer_norm:
            x = self.norm1(x)

        if self.use_residual:
            x = x + self.residual1(identity)

        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 保存第一层输出用于跳跃连接
        skip_input = x

        # 第二层GAT
        x = self.conv2(x, edge_index)

        if self.use_layer_norm:
            x = self.norm2(x)

        if self.use_residual:
            x = x + self.residual2(skip_input)

        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 跳跃连接
        if self.use_skip_connection:
            skip = self.skip_connection(skip_input)
            x = x + skip

        # 输出层
        x = self.out_layer(x)

        # 如果提供了batch信息，进行图池化
        if batch is not None:
            x = global_mean_pool(x, batch)

        return x
