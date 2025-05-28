import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        """
        图注意力网络模型
        
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出维度
            heads: 注意力头数
            dropout: Dropout比率
        """
        super(GATModel, self).__init__()
        self.dropout = dropout
        
        # 第一层GAT
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # 第二层GAT
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        # 输出层
        self.out_layer = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x: 节点特征矩阵
            edge_index: 边索引矩阵
        """
        # 第一层GAT
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层GAT
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 输出层
        x = self.out_layer(x)
        
        return x 