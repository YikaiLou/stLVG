from typing import List, Optional, Union, Any


import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing

from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from .disandang import generate_spatial_distance_graph,theta_from_spatial_graph
from .csr_operations import remove_greater_than, row_normalize

def sym_norm(edge_index:torch.Tensor,
             num_nodes:int,
             edge_weight:Optional[Union[Any,torch.Tensor]]=None,
             improved:Optional[bool]=False,
             dtype:Optional[Any]=None
    )-> List:
    r"""
    Replace `GCNConv.norm` from https://github.com/mengliu1998/DeeperGNN/issues/2
    """
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)#zihuan

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0#计算度矩阵

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

#修改message方法
def gaussian_weight_1d(distance: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Calculate normalized gaussian value for a given distance from central point
    Normalized by root(2*pi) x sigma
    """
    # 将 sigma 参数转换为 torch.Tensor 类型
    sigma_tensor = torch.tensor(sigma, dtype=distance.dtype)
    # 所有操作都使用张量
    two_pi_tensor = torch.tensor(2.0) * torch.pi  # 创建一个张量来存储 2 * pi
    return torch.exp(-0.5 * distance.pow(2) / sigma_tensor.pow(2)) / (sigma_tensor * torch.sqrt(two_pi_tensor))


class CombGaussian(MessagePassing):
    r"""
    LGCN (GCN without learnable and concat)
    
    Parameters
    ----------
    K
        K-hop neighbor to propagate
    sigma
        Standard deviation for Gaussian weight calculation
    """

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        edge_index, norm = sym_norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        gaussian_norm = gaussian_weight_1d(norm, self.sigma)  # 计算高斯权重

        xs = [x]
        for k in range(self.K):
        # 应用高斯权重并传递消息
            message = gaussian_norm * xs[-1]  # 确保这里的乘法是逐元素的
        xs.append(self.propagate(edge_index, x=message, norm=norm))

        return torch.cat(xs, dim=1)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        # Simply return the features as the message
        return x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)