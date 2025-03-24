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
                 K_1: int,
                 sigma: Optional[float] = 1.0,
                 cached: Optional[bool] = False,
                 bias: Optional[bool] = True,
                 **kwargs):
        super(CombBanksy, self).__init__(aggr='add', **kwargs)
        self.K_1 = K_1
        self.sigma = sigma

    
        
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
        return x
        # return torch.stack(xs, dim=0).mean(dim=0)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                        #  self.in_channels, self.out_channels,
                                         self.K)
    
