from typing import List, Optional, Tuple, Union, Any


import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing

from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from .disandang import generate_spatial_distance_graph,theta_from_spatial_graph
from .csr_operations import remove_greater_than, row_normalize


def euclidean_distance(coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    # 获取边的数量
    num_edges = edge_index.size(1)
    
    # 初始化一个大小为 [num_edges] 的张量来存储每一条边的距离
    distances = torch.zeros(num_edges)
    
    # 计算每一条边的距离并存储在 distances 中
    for i in range(num_edges):
        start_node = edge_index[0, i]
        end_node = edge_index[1, i]
        # 获取起始节点和结束节点的坐标
        start_coord = coords[start_node]
        end_coord = coords[end_node]
        # 计算欧几里得距离
        distances[i] = torch.norm(start_coord - end_coord)
    
    return distances

def sym_norm(edge_index: torch.Tensor,
             num_nodes: int,
             coordinates: torch.Tensor,  # 添加节点坐标作为参数
             edge_weight:Optional[Union[Any,torch.Tensor]]=None,
             improved: Optional[bool] = False,
             dtype: Optional[Any] = None
             ) -> List:
    # 假设 coordinates 是一个形状为 [2, n] 的 NumPy 数组，其中 n 是节点的数量
    # coordinates = np.array([[x1, x2, ..., xn],
    #                         [y1, y2, ..., yn]])

    # print("Coordinates Size:", coordinates.size())
    # edge_index 是一个形状为 [2, E] 的张量，包含边的索引
    # edge_index = torch.tensor([[i1, i2, ..., iE], [j1, j2, ..., jE]])

    # 调用函数
    # edge_index_normalized, edge_weight_normalized = sym_norm(edge_index, num_nodes, coordinates)
    

    if edge_weight is None:
    # 计算边权重，即两点之间的欧几里得距离
        edge_distance = euclidean_distance(coordinates, edge_index)
        edge_weight = gaussian_weight_1d(edge_distance, sigma = 1.0)
        edge_weight *= 100
        print("edge_weight:", edge_weight)
        print("edge_distance:", edge_distance)
    
    # 如果需要，添加自环
    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
    
    # 计算节点的度
    # row, col = edge_index
    # deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    # #print("Degree before inverse square root:", deg)
    # deg_inv_sqrt = deg.pow(-0.5)
    

    # 避免除以零的错误
    # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # 计算对称归一化的边权重
    # norm_edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    return edge_index, edge_weight

def inverse_tensor(tensor: torch.Tensor) -> torch.Tensor:
    
    not_zero_mask = tensor.nonzero(as_tuple=False)
    
    # 使用掩码选择非零元素，计算它们的倒数
    # 然后将这些倒数赋值回原始张量的相应位置
    # 零值保持不变，因为没有被选择
    tensor[not_zero_mask] = 1.0 / tensor[not_zero_mask]
    
    return tensor


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


class CombBanksy(MessagePassing):
    r"""
    LGCN (GCN without learnable and concat)
    
    Parameters
    ----------
    in_channels
        input size
    out_channels
        output size
    K
        K-hop neighbor to propagate
    hindden_channels
        hindden size
    sigma
        Gaussian decay parameters
    """
    def __init__(self, coordinates: torch.Tensor, K: int = 1,
                 sigma: float = 1.0, mlp_layers: List[int] = None, **kwargs):
        super(CombBanksy, self).__init__(aggr='add', **kwargs)
        self.coordinates = coordinates
        self.sigma = sigma
        # 添加一个MLP层，层数和尺寸由mlp_layers指定
        self.mlp = nn.Sequential()
        if mlp_layers is not None:
            in_features = coordinates.size(1)  # 假设coordinates是特征维度
            for out_features in mlp_layers:
                self.mlp.add_module('linear', nn.Linear(in_features, out_features))
                in_features = out_features
                if out_features != mlp_layers[-1]:  # 如果不是最后一层，添加ReLU激活函数
                    self.mlp.add_module('relu', nn.ReLU())
        
    def forward(self, x:torch.Tensor,
                edge_index:torch.Tensor,
                
                edge_weight:Union[torch.Tensor,None]=None):
        # print("Input x shape:", x.shape)
        # print("Input edge_index shape:", edge_index.shape)
        # print("Input edge_weight shape:", edge_weight.shape if edge_weight is not None else None)

        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight,
        #                                 dtype=x.dtype)
        edge_index, norm = sym_norm(edge_index, x.size(0), edge_weight,
                                        dtype=x.dtype)
        print(norm)
        xs = [x]
        for k in range(self.K_1):
            xs.append(self.propagate(edge_index, x=xs[-1], norm=norm))#传递
        x = torch.cat(xs, dim = 1)#拼接
        x = self.mlp(x)
        return x
        # return torch.stack(xs, dim=0).mean(dim=0)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                        #  self.in_channels, self.out_channels,
                                         self.K)
    




# import torch.nn.functional as F

# class CombUnweighted(MessagePassing):
#     r"""
#     LGCN (GCN without learnable weights and concat) with integrated MLP
    
#     Parameters
#     ----------
#     K : int, optional
#         K-hop neighbor to propagate, by default 1
#     hidden_size : int, optional
#         hidden size of MLP, by default 512
#     output_size : int, optional
#         output size of MLP, by default same as input_size
#     dropout : float, optional
#         dropout ratio, by default 0.2
#     cached : bool, optional
#         whether to cache the computation, by default False
#     bias : bool, optional
#         whether to add bias, by default True
#     """
#     def __init__(self, K: int = 1, 
#                  input_size:Optional[int] = None,
#                  hidden_size: int = 512, 
#                  output_size: Optional[int] = None, 
#                  dropout: float = 0.2, 
#                  cached: bool = False, 
#                  bias: bool = True, **kwargs):
#         super(CombUnweighted, self).__init__(aggr='add', **kwargs)
        
#         self.K = K
#         self.mlp = nn.Sequential(
#             nn.Linear(K * input_size, hidden_size),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size, output_size or input_size),
#         )
#         self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        
#         edge_index, norm = sym_norm(edge_index, x.size(0), edge_weight,
#                                         dtype=x.dtype)

#         xs = [x]
#         for k in range(self.K_1):
#             xs.append(self.propagate(edge_index, x=xs[-1], norm=norm))#传递
#         x = torch.cat(xs, dim = 1)#拼接
#         x = self.mlp(x)
#         return x

#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j

#     def __repr__(self):
#         return '{}(K={}, hidden_size={})'.format(self.__class__.__name__, self.K, self.hidden_size)
























# #修改message方法
# def gaussian_weight_1d(distance: float, sigma: float):
#     """
#     Calculate normalized gaussian value for a given distance from central point
#     Normalized by root(2*pi) x sigma
#     """
#     return np.exp(-0.5 * distance ** 2 / sigma ** 2) / (sigma * np.sqrt(2 * np.pi))

# def gaussian_weight_2d(distance: float, sigma: float):
#     """
#     Calculate normalized gaussian value for a given distance from central point
#     Normalized by 2*pi*sigma-squared
#     """
#     sigma_squared = float(sigma) ** 2
#     return np.exp(-0.5 * distance ** 2 / sigma_squared) / (sigma_squared * 2 * np.pi)
# #首先，定义高斯衰减计算权重的方法

# # def theta_from_spatial_graph(locations: np.ndarray,
# #                              spatial_graph: csr_matrix,
# #                              ):
# #     """
# #     get azimuthal angles from spatial graph and coordinates
# #     (assumed dim 1: x, dim 2: y, dim 3: z...)

# #     returns CSR matrix with theta (azimuthal angles) as .data
# #     """

# #     theta_data = np.zeros_like(spatial_graph.data, dtype=np.float32)

# #     for n in range(spatial_graph.indptr.shape[0] - 1):
# #         ptr_start, ptr_end = spatial_graph.indptr[n], spatial_graph.indptr[n + 1]
# #         nbr_indices = spatial_graph.indices[ptr_start:ptr_end]

# #         self_coord = locations[[n], :]
# #         nbr_coord = locations[nbr_indices, :]
# #         relative_coord = nbr_coord - self_coord

# #         theta_data[ptr_start:ptr_end] = np.arctan2(
# #             relative_coord[:, 1], relative_coord[:, 0])

# #     theta_graph = spatial_graph.copy()
# #     theta_graph.data = theta_data

# #     return theta_graph
# # #定义角度矩阵

# class CombGaussian(MessagePassing):
#     r"""
#     LGCN (GCN without learnable and concat)
    
#     Parameters
#     ----------
#     K
#         K-hop neighbor to propagate
#     sigma
#         Standard deviation for Gaussian weight calculation
#     """
        


#     def __init__(self, 
#                  K_1:Optional[int]=1,
#                  K_2:Optional[int]=1,
#                  sigma: float = 1.0,
#                  cached:Optional[bool]=False,
#                  bias:Optional[bool]=True,
#                  **kwargs):
#         super(CombBanksy, self).__init__(aggr='mean', **kwargs)
#         self.K_1 = K_1
#         self.K_2 = K_2
#         self.sigma = sigma


#     def forward(self, x: torch.Tensor,
#                 edge_index: torch.Tensor,
#                 edge_weight: Union[torch.Tensor, None] = None):
        
#         edge_index, norm = sym_norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)

#         xs = [x]
#         for k in range(self.K_1):
#             xs.append(self.propagate(edge_index, x=xs[-1], norm=norm))  # 传递
        
#         for k in range(self.K_2):
#             xs.append(self.propagate(edge_index, x=xs[-1], norm=norm))

#         return torch.cat(xs, dim=1)  # 拼接

#     def message(self, x_j, norm):
#         return  norm.view(-1, 1) * x_j*gaussian_weight_1d(norm, self.sigma)  # 使用高斯权重进行消息传递
    

#     def __repr__(self):
#         return '{}({}, {}, K={})'.format(self.__class__.__name__, self.K, self.sigma)
    






    
