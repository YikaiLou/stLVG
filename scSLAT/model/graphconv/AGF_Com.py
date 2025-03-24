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
             x: torch.Tensor, 
             coordinates: torch.Tensor,  # 添加节点坐标作为参数
             m: int, #传入m
             edge_weight:Optional[Union[Any,torch.Tensor]]=None,
             improved: Optional[bool] = False,
             dtype: Optional[Any] = None
             ) -> List:
    # coordinates[0] 是一个形状为 [n, 2] 的张量，其中 n 是节点的数量
    # coordinates[0] = torch.tensor([[x1, x2, ..., xn],
    #                               [y1, y2, ..., yn]])

    # edge_index[0] 是一个形状为 [2, E] 的张量，包含两个slice的边索引
    # 此时相比于同样的k, 经过AGF变换, 边的num_neighbor变为2k
    # edge_index[0] = torch.tensor([[i1, i2, ..., iE], 
    #                              [j1, j2, ..., jE]])
    # x_arr = x.cpu().numpy()
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
    # indptr, data = graph_out.indptr, graph_out.data

    # for n in range(len(indptr) - 1):
    #         start_ptr, end_ptr = indptr[n], indptr[n + 1]
    #         if end_ptr >= start_ptr:
    #             nbrs = data[start_ptr:end_ptr]
    #             median_r = np.median(nbrs)
    #             weights = np.exp(-(nbrs / median_r) ** 2)
    #             data[start_ptr:end_ptr] = weights

    # print(graph_out)

    graph_out = row_normalize(graph_out, verbose=False)

    # print(graph_out)

    tensor_data_agf = torch.tensor(graph_out.data, dtype=torch.float32)

    if edge_weight is None:
        edge_weight = tensor_data_agf
    
    # 如果需要，添加自环
    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
    
    # print(edge_weight.size())
    # print(distances)

    return edge_index, edge_weight


# def azimuthal_gaussian_transform(x: torch.Tensor, 
#                                  edge_index: torch.Tensor, 
#                                  coor: torch.Tensor, 
#                                  m: int,
#                                  edge_weight: Optional[Union[Any,torch.Tensor]]=None,
#                                  sigma: float = 1.0
#                                  ) -> torch.Tensor:
    
#         row, col = edge_index
#         num_nodes = x.size(0)

#         # Step 1: Calculate Gaussian weight and complex exponential
#         distances = torch.norm(coor[row] - coor[col], dim=1)
#         spatial_graph = csr_matrix((distances, (row, col)), shape=(num_nodes, num_nodes))
#         graph_out = row_normalize(spatial_graph, verbose=False) # [N,F]

#         # Extract 1D array of normalized data
#         one_dimensional_array = np.array(spatial_graph.data)
#         normalized_tensor = torch.tensor(one_dimensional_array, dtype=torch.float32)

#         # Compute Gaussian weight Γ_uv
#         edge_weight = gaussian_weight_1d(normalized_tensor, sigma) # [E,]

#         # Compute complex exponential e^(iφ_uv)
#         theta_graph = theta_from_spatial_graph(coor.numpy(), spatial_graph)
#         graph_out.data = graph_out.data * np.exp(1j * m * theta_graph.data)
#         complex_array = np.array(graph_out.data)
#         complex_array_tensor = torch.tensor(complex_array, dtype=torch.float32)

#         # Step 2: Compute weighted neighborhood mean (bar_g_uq)
#         weighted_neigh = x[col] * edge_weight.view(-1, 1) # [E,F]
#         neigh_mean = torch.zeros_like(x)
#         neigh_mean.index_add_(0, row, weighted_neigh)

#         norm_mean = neigh_mean.norm(dim=1, keepdim=True)
#         neigh_mean = neigh_mean / norm_mean  # This is bar_g_uq

#         # Step 3: Compute centered features (g_vq - bar_g_uq)
#         centered_x = x[col] - neigh_mean[row]

#         # Step 4: Compute AGF matrix element G~_uq
#         weighted_x = centered_x * edge_weight.view(-1, 1) * complex_array_tensor.view(-1, 1) # [E,F]

#         # Step 5: Accumulate to get final output
#         gradient_matrix = torch.zeros((num_nodes, x.size(1)), dtype=x.dtype)
#         gradient_matrix.index_add_(0, row, weighted_x) # [N,F]

#         # Step 6: Take magnitude to get final AGF matrix
#         final_output = torch.abs(gradient_matrix)
        
#         return final_output

# def azimuthal_gaussian_transform(
#     x: torch.Tensor,
#     edge_index: torch.Tensor,
#     coor: torch.Tensor,
#     m: int,
#     edge_weight: Optional[torch.Tensor] = None,
#     sigma: float = 1.0
# ) -> torch.Tensor:
#     row, col = edge_index
#     num_nodes = x.size(0)

#     # Step 1: 计算复数权重 Γ_uv e^{i mφ_uv}（保留复数类型）
#     distances = torch.norm(coor[row] - coor[col], dim=1)
    
#     # 构建稀疏矩阵（兼容设备）
#     row_np = row.cpu().numpy()
#     col_np = col.cpu().numpy()
#     spatial_graph = csr_matrix((distances.cpu().numpy(), (row_np, col_np)), 
#                               shape=(num_nodes, num_nodes))
#     graph_out = row_normalize(spatial_graph)  # 假设返回复数权重
    
#     # 转换为PyTorch复数张量，并匹配设备
#     complex_weights = torch.tensor(graph_out.data, dtype=torch.complex64).to(x.device)
#     edge_weight_abs = torch.abs(complex_weights)  # 高斯权重 Γ_uv

#     # Step 2: 计算未归一化的加权邻居均值
#     weighted_neigh = x[col] * edge_weight_abs.view(-1, 1)
#     neigh_mean = torch.zeros_like(x)
#     neigh_mean.index_add_(0, row, weighted_neigh)

#     # Step 3: 计算中心化特征
#     centered_x = x[col] - neigh_mean[row]

#     # Step 4: 复数乘法并取模
#     weighted_x_complex = centered_x.to(torch.complex64) * complex_weights.view(-1, 1)  # [E,F] 复数
#     weighted_x_abs = torch.abs(weighted_x_complex)  # 取模 [E,F] 浮点数

#     # Step 5: 聚合结果
#     gradient_matrix = torch.zeros((num_nodes, x.size(1)), dtype=x.dtype, device=x.device)
#     gradient_matrix.index_add_(0, row, weighted_x_abs)  # 累加模值
#     final_output = gradient_matrix  # 最终结果无需再取绝对值（已取模）

#     return final_output

import torch
from torch import Tensor
from typing import Optional

def azimuthal_gaussian_transform(
    x: Tensor,
    edge_index: Tensor,
    coor: Tensor,
    m: int,
    edge_weight: Optional[Tensor] = None,
    sigma: float = 1.0,
    rotation_factor: int = 2  # 控制旋转角度倍数，默认2倍
) -> Tensor:
    """
    基于复数权重和旋转角度差的方位高斯变换
    
    参数:
    - x: 节点特征矩阵 [N, F]
    - edge_index: 边索引 [2, E]
    - coor: 节点坐标矩阵 [N, D]
    - m: 基础谐波阶数
    - edge_weight: 预定义边权重 [E,]
    - sigma: 高斯核标准差
    - rotation_factor: 旋转角度倍数（例如2表示角度翻倍）
    
    返回:
    - output: 聚合后的方向敏感特征 [N, F]
    """
    row, col = edge_index
    num_nodes, feat_dim = x.shape

    # ==============================
    # Step 1. 计算高斯权重 Γ_uv
    # ==============================
    delta_coor = coor[col] - coor[row]  # 目标节点坐标 - 源节点坐标 [E, D]
    distances = torch.norm(delta_coor, dim=1)  # [E,]
    
    if edge_weight is None:
        edge_weight = torch.exp(-distances**2 / (2 * sigma**2))  # [E,]

    # ==============================
    # Step 2. 计算复数相位
    # ==============================
    # 计算方位角 φ ∈ [0, 2π)
    phi = torch.atan2(delta_coor[:, 1], delta_coor[:, 0])  # [E,]
    phi = torch.where(phi < 0, phi + 2 * torch.pi, phi)  # 转换到 [0, 2π)
    
    # 原始复数相位
    base_phase = torch.exp(1j * m * phi)  # [E,]
    complex_weights = edge_weight * base_phase  # [E,]

    # ==============================
    # Step 3. 计算旋转后相位
    # ==============================
    rotated_phase = torch.exp(1j * m * rotation_factor * phi)  # [E,]
    rotated_weights = edge_weight * rotated_phase  # [E,]

    # ==============================
    # Step 4. 复数加权邻居特征
    # ==============================
    # 将特征转换为复数类型 [E, F]
    x_complex = x[col].to(torch.complex64)
    weighted_neigh = x_complex * complex_weights.view(-1, 1)  # [E, F]

    # ==============================
    # Step 5. 计算旋转权重差异
    # ==============================
    # 广播旋转权重到特征维度 [E, F]
    rotated_weights_expanded = rotated_weights.view(-1, 1).expand(-1, feat_dim)
    diff_complex = weighted_neigh - rotated_weights_expanded  # [E, F]

    # ==============================
    # Step 6. 取模并聚合结果
    # ==============================
    diff_abs = torch.abs(diff_complex)  # [E, F]
    output = torch.zeros((num_nodes, feat_dim), dtype=x.dtype, device=x.device)
    output.index_add_(0, row, diff_abs)  # [N, F]

    return output

# def create_spatial_graph(edge_index, edge_weight, num_nodes):
#     # 根据 edge_index 和 edge_weight 创建一个稀疏矩阵表示的空间图
#     row, col = edge_index
#     spatial_graph = csr_matrix((edge_weight.cpu().numpy(), (row.cpu().numpy(), col.cpu().numpy())), shape=(num_nodes, num_nodes))
#     return spatial_graph

# def cap_gradient(edge_index: torch.Tensor,
#                  num_nodes: int,
#                  x: torch.Tensor, 
#                  coordinates: torch.Tensor, 
#                  m: int, 
#                  edge_weight: Optional[Union[Any,torch.Tensor]]=None,
#                  improved: Optional[bool] = False,
#                  dtype: Optional[Any] = None
#                 ) -> List:

#     coor_arr = coordinates.cpu().numpy()  
#     edge_index_arr = edge_index.cpu().numpy() 
#     start_points = edge_index_arr[0]
#     end_points = edge_index_arr[1]
#     num_nodes = np.max(edge_index_arr) + 1  
#     distances = np.linalg.norm(coor_arr[start_points] - coor_arr[end_points], axis=1)
#     spatial_graph = csr_matrix((distances, (start_points, end_points)), shape=(num_nodes, num_nodes))

#     # 根据距离构建权重矩阵
#     graph_out = spatial_graph.copy()
#     indptr, data = graph_out.indptr, graph_out.data

#     for n in range(len(indptr) - 1):

#             start_ptr, end_ptr = indptr[n], indptr[n + 1]
#             if end_ptr >= start_ptr:
#                 nbrs = data[start_ptr:end_ptr]
#                 median_r = np.median(nbrs)
#                 weights = np.exp(-(nbrs / median_r) ** 2)
#                 data[start_ptr:end_ptr] = weights

#     graph_out = row_normalize(graph_out, verbose=False)
#     # Azimuthal Fourier Transform
#     m = m
    
#     if m > 0 :
#         theta_graph = theta_from_spatial_graph(coor_arr, spatial_graph)
#         graph_out.data = graph_out.data * np.exp(1j * m * theta_graph.data)

#         weights_abs = graph_out.copy()
#         weights_abs.data = np.absolute(weights_abs.data)

#         nbr_avgs = torch.from_numpy(weights_abs @ x.cpu().numpy()).to(x.device)

#         # print(nbr_avgs.size())
#         nbr_mat = np.zeros(x.shape)
        
#         for n in range(graph_out.shape[0]):
#             start_ptr, end_ptr = graph_out.indptr[n], graph_out.indptr[n + 1]
#             ind_temp = graph_out.indices[start_ptr:end_ptr]
#             weight_temp = torch.tensor(graph_out.data[start_ptr:end_ptr], dtype=torch.float32, device=x.device)
#             zerod = x[ind_temp, :] - nbr_avgs[n, :]
#             nbr_mat[n, :] = torch.abs(weight_temp.unsqueeze(0) @ zerod).cpu().numpy()

#         graph_out = torch.tensor(nbr_mat, dtype=torch.float32, device=x.device)
#     else:
#         theta_graph = None

#     if edge_weight is None:
#         edge_weight = graph_out

#     return edge_weight


def gaussian_weight_1d(distance: torch.Tensor, sigma: float) -> torch.Tensor:

    sigma_tensor = torch.tensor(sigma, dtype=distance.dtype)
    two_pi_tensor = torch.tensor(2.0) * torch.pi  
    return torch.exp(-0.5 * distance.pow(2) / sigma_tensor.pow(2)) / (sigma_tensor * torch.sqrt(two_pi_tensor))


class CombGaussian_F(MessagePassing):
    r"""
    LGCN (GCN without learnable and concat)
    
    Parameters
    ----------
    coordinates
        the coordinates of the points
    m
        the azimuthal transform order
    K
        K-hop neighbor to propagate
    sigma
        Standard deviation for Gaussian weight calculation
    """
    def __init__(self, 
                 coordinates: torch.Tensor,
                 m: int,
                 K: int = 1, 
                 sigma: float = 1.0,
                 **kwargs):
        super(CombGaussian_F, self).__init__(aggr='add', **kwargs)
        self.coordinates = coordinates
        self.m = m
        self.K = K
        self.sigma = sigma

    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor, 
                coor: torch.Tensor,
                m: int,
                edge_weight: Optional[torch.Tensor] = None):
        xs = [x]
        # norm = None
        if m == 0 :
            edge_index, norm = sym_norm(edge_index, x.size(0), x, coor, m, edge_weight, dtype=x.dtype)
            for k in range(self.K):
                message = self.propagate(edge_index, x=xs[-1], norm = norm)
                xs.append(message) 
            
        elif m > 0:
            final_output = azimuthal_gaussian_transform(x, edge_index, coor, m)
            
            xs.append(final_output)            
             
        return torch.cat(xs, dim=1) #, norm

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        if self.m == 0:
            return x_j * gaussian_weight_1d(norm, self.sigma).view(-1, 1)
        elif self.m > 0:
            # return x_j * gaussian_weight_1d(norm, self.sigma).view(-1, 1)
            return x_j
        # return x_j * norm.view(-1, 1)
    
    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
    



def theta_from_spatial_graph(locations: np.ndarray, spatial_graph: csr_matrix):
    
    theta_data = np.zeros_like(spatial_graph.data, dtype=np.float32)

    for n in range(spatial_graph.indptr.shape[0] - 1):
        ptr_start, ptr_end = spatial_graph.indptr[n], spatial_graph.indptr[n + 1]
        nbr_indices = spatial_graph.indices[ptr_start:ptr_end]

        self_coord = locations[n, :]
        nbr_coord = locations[nbr_indices, :]
        relative_coord = nbr_coord - self_coord

        theta_data[ptr_start:ptr_end] = np.arctan2(relative_coord[:, 1], relative_coord[:, 0])

    theta_graph = spatial_graph.copy()
    theta_graph.data = theta_data

    return theta_graph


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






