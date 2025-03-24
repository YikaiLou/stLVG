from typing import List, Optional, Tuple, Union, Any


import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing

from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from .disandang import generate_spatial_distance_graph,theta_from_spatial_graph
from .csr_operations import remove_greater_than, row_normalize


# 在utils中, 传递的参数是两个slice的features, edges, coordinates 组成的List
# 随后的embedding中, 会分别获取List中的数据，这些数据以tensor形式储存


def sym_norm(edge_index: torch.Tensor,
             num_nodes: int,
             coordinates: torch.Tensor,  # 添加节点坐标作为参数
             edge_weight:Optional[Union[Any,torch.Tensor]]=None,
             improved: Optional[bool] = False,
             dtype: Optional[Any] = None
             ) -> List:
    # coordinates[0] 是一个形状为 [n, 2] 的张量，其中 n 是节点的数量
    # coordinates[0] = torch.tensor([[x1, x2, ..., xn],
    #                               [y1, y2, ..., yn]])

    # edge_index[0] 是一个形状为 [2, E] 的张量，包含两个slice的边索引
    # edge_index[0] = torch.tensor([[i1, i2, ..., iE], 
    #                              [j1, j2, ..., jE]])

    coor_arr = coordinates.cpu().numpy()  
    edge_index_arr = edge_index.cpu().numpy()   # 确保tensor在CPU上, 并转换为 numpy 数组
    start_points = edge_index_arr[0]
    end_points = edge_index_arr[1]
    # 构建 CSR 矩阵, 值为两点间的距离
    num_nodes = np.max(edge_index_arr) + 1  
    distances = np.linalg.norm(coor_arr[start_points] - coor_arr[end_points], axis=1)
    spatial_graph = csr_matrix((distances, (start_points, end_points)), shape=(num_nodes, num_nodes))

    # 根据距离构建权重矩阵

    graph_out = spatial_graph.copy()

    # 高斯衰减函数, 得到权重
    indptr, data = graph_out.indptr, graph_out.data

    for n in range(len(indptr) - 1):

            start_ptr, end_ptr = indptr[n], indptr[n + 1]
            if end_ptr >= start_ptr:
                nbrs = data[start_ptr:end_ptr]
                median_r = np.median(nbrs)
                weights = np.exp(-(nbrs / median_r) ** 2)
                data[start_ptr:end_ptr] = weights

    # 归一化
    graph_out = row_normalize(graph_out, verbose=False)

    tensor_data = torch.tensor(data, dtype=torch.float32)

    if edge_weight is None:
        edge_weight = tensor_data

    # 如果需要，添加自环
    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)


    return edge_index, edge_weight


def gaussian_weight_1d(distance: torch.Tensor, sigma: float) -> torch.Tensor:
    
    # 使用张量操作
    sigma_tensor = torch.tensor(sigma, dtype=distance.dtype)
    two_pi_tensor = torch.tensor(2.0) * torch.pi  
    return torch.exp(-0.5 * distance.pow(2) / sigma_tensor.pow(2)) / (sigma_tensor * torch.sqrt(two_pi_tensor))


class CombGaussian(MessagePassing):
    r"""
    LGCN (GCN without learnable and concat)
    
    Parameters
    ----------
    coordinates
        the coordinates of the points
    K
        K-hop neighbor to propagate
    sigma
        Standard deviation for Gaussian weight calculation
    """
    def __init__(self, 
                 coordinates :torch.Tensor,
                 K: int = 1, sigma: float = 1.0, **kwargs):
        super(CombGaussian, self).__init__(aggr='add', **kwargs)
        self.coordinates = coordinates
        self.K = K
        self.sigma = sigma

    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor, 
                coor : torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None):
        edge_index, norm = sym_norm(edge_index, x.size(0), coor , edge_weight, dtype=x.dtype)
        xs = [x]
        for k in range(self.K):
            message = self.propagate(edge_index, x=xs[-1], norm=norm)
            xs.append(message)

        return torch.cat(xs, dim=1)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:

        # return x_j * gaussian_weight_1d(norm, self.sigma).view(-1, 1)
        return x_j * norm.view(-1, 1)
    
    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
    


# (Optional)
def spatial_graph(edges:torch.Tensor):
    # 获取边的起点和终点
    start_points = edges[0].numpy()
    end_points = edges[1].numpy()

    # 获取节点的总数
    num_nodes = max(np.max(start_points), np.max(end_points)) + 1  # 假设节点编号是从0开始连续的

    # 构建CSR矩阵
    data = np.ones(len(start_points))
    spatial_graph = csr_matrix((data, (start_points, end_points)), shape=(num_nodes, num_nodes))
    return spatial_graph






