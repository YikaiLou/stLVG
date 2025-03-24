from typing import List, Optional, Union, Any


import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing

from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from .disandang import generate_spatial_distance_graph,theta_from_spatial_graph
from .csr_operations import remove_greater_than, row_normalize

def sym_norm(locations: np.ndarray,
             edge_index:torch.Tensor,
             num_nodes:int,
             edge_weight:Optional[Union[Any,torch.Tensor]]=None,
             improved:Optional[bool]=False,
             dtype:Optional[Any]=None,
             verbose: bool = True
    )-> List:
    r"""
    Replace `GCNConv.norm` from https://github.com/mengliu1998/DeeperGNN/issues/2
    """
    graph_out = generate_spatial_distance_graph(locations)
    #首先根据距离生成邻接矩阵（值为两点间距离）
    if edge_weight is None:
        #通过距离生成权重矩阵
        indptr, data = graph_out.indptr, graph_out.data

        for n in range(len(indptr) - 1):

            start_ptr, end_ptr = indptr[n], indptr[n + 1]
            if end_ptr >= start_ptr:
                # row entries correspond to a cell's neighbours
                nbrs = data[start_ptr:end_ptr]
                median_r = np.median(nbrs)
                #### Changed here
                weights = np.exp(-(nbrs / median_r) ** 2)
                data[start_ptr:end_ptr] = weights

    graph_out = row_normalize(graph_out, verbose=verbose)


    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)#自环

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0#计算度矩阵

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class CombBanksy(MessagePassing):
    r"""
    LGCN (GCN without learnable and concat)
    
    Parameters
    ----------
    K
        K-hop neighbor to propagate
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 K_1: int,
                 hidden_channels: int,
                 sigma: Optional[float] = 1.0,
                 cached: Optional[bool] = False,
                 bias: Optional[bool] = True,
                 **kwargs):
        super(CombBanksy, self).__init__(aggr='add', **kwargs)
        self.K_1 = K_1
        self.sigma = sigma
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, x:torch.Tensor,
                edge_index:torch.Tensor,
                
                edge_weight:Union[torch.Tensor,None]=None):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight,
        #                                 dtype=x.dtype)
        edge_index, norm = sym_norm(edge_index, x.size(0), edge_weight,
                                        dtype=x.dtype)

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
    






    
