import torch
from torch_geometric.nn import MessagePassing

import torch


# 假设我们有一些随机生成的节点特征
num_nodes = 5  # 假设有5个节点
features = torch.rand((num_nodes, 4))  # 每个节点有4个特征


# 假设我们有以下的边列表，每条边由两个节点索引组成
edges = torch.tensor([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 0]  # 假设图是一个环
], dtype=torch.long)

# 确保 edges 是一个二维张量
if edges.dim() == 1:
    edges = edges.view(-1, 2)

# 将边列表转换为边索引
edge_index = edges.t().contiguous()

# 确保 edge_index 的数据类型为长整型
edge_index = edge_index.long()

print(edge_index)

# 随机生成的边权重，与边的总数E相同
edge_weight = torch.rand(edge_index.size(1))

def gaussian_weight(distance, sigma):
    # 高斯权重函数，根据节点间距离计算权重
    return torch.exp(-(distance / sigma) ** 2)

class SimpleGCN(MessagePassing):
    def __init__(self, sigma=1.0):
        super(SimpleGCN, self).__init__(aggr='add')
        self.sigma = sigma

    def forward(self, x, edge_index, edge_weight):
        # 这里我们传递edge_weight到propagate方法
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return x

    def message(self, x_j, edge_weight):
    # 计算高斯权重并乘以邻居的特征
    # 确保 edge_weight 是一个一维张量，并且可以与 x_j 正确广播
        gauss_weight = torch.exp(- (edge_weight.unsqueeze(-1) ** 2) / self.sigma ** 2)
        return gauss_weight * x_j

# 创建SimpleGCN层的实例
gcn = SimpleGCN(sigma=1.0)

# 执行图卷积操作
x = gcn(features, edge_index, edge_weight)

print(x)